# EnhancedTensorBoardHook 코드 리뷰

## 발견된 주요 문제점

### 1. Performance Overhead (성능 오버헤드)

#### ❌ 문제점
- **`after_train_iter`에서 Learning Rate를 매 iteration마다 로깅** (Line 727-728)
  - `self.log_lr`가 True이면 interval 체크 없이 매번 실행됨
  - 불필요한 TensorBoard write 발생

- **Training image 로깅 시 매번 모델 재추론** (Line 385-406)
  - `model.test_step()`을 매번 호출하여 GPU 연산 추가
  - `train_image_interval`이 적용되지만, 실행될 때마다 큰 오버헤드

#### ✅ 해결 방안
```python
# Learning rate는 interval을 두고 로깅해야 함
if self.log_lr and runner.iter % self.grad_norm_interval == 0:  # 같은 interval 사용
    self._log_learning_rate(runner)
```

---

### 2. Memory Safety (메모리 안전성)

#### ❌ 문제점

**a) First batch data를 CPU에 복사하여 메모리에 계속 유지** (Line 714-716)
```python
self._first_batch_data = {
    'inputs': inputs.detach().cpu().clone(),  # 전체 배치를 CPU 메모리에 복사
    'data_samples': data_batch['data_samples']
}
```
- Batch size가 크면 수백 MB의 CPU 메모리 점유
- 학습 내내 메모리에 유지됨

**b) Gradient norm 계산 시 불필요한 `.item()` 호출** (Line 180-188)
```python
param_norm = param.grad.data.norm(2)
total_norm += param_norm.item() ** 2  # GPU -> CPU 동기화
param_norms[layer_name] = param_norm.item()  # 또 동기화
```
- 각 레이어마다 GPU-CPU 동기화 발생
- 성능 저하 유발

**c) Image denormalization에서 tensor clone 후 in-place 연산** (Line 285-288)
```python
img = img_tensor.clone()
for t, m, s in zip(img, mean, std):
    t.mul_(s).add_(m)  # in-place 연산
```
- Clone은 했지만 여전히 위험한 패턴
- 원본이 필요 없으면 clone 불필요

#### ✅ 해결 방안
```python
# 1. First batch는 작은 subset만 저장
self._first_batch_data = {
    'inputs': inputs[:4].detach().cpu().clone(),  # 4개만 저장
    'data_samples': data_batch['data_samples'][:4]
}

# 2. Gradient norm은 tensor로 계산 후 한번만 CPU 이동
norms_list = []
for name, param in model.named_parameters():
    if param.grad is not None:
        norms_list.append(param.grad.data.norm(2))

if norms_list:
    norms_tensor = torch.stack(norms_list)
    total_norm = norms_tensor.norm(2).item()  # 한 번만 동기화
```

---

### 3. DDP 호환성

#### ❌ 문제점
- **DDP wrapper 처리가 없음**
  - `runner.model`은 DDP일 때 `DistributedDataParallel` 객체
  - 실제 모델은 `runner.model.module`에 있음
  - 현재 코드는 DDP에서 `module` 없이 직접 접근

#### ✅ 해결 방안
```python
def _get_model(self, runner: Runner):
    """Get the actual model, unwrapping DDP if necessary."""
    model = runner.model
    # Unwrap DDP/FSDP wrapper
    if hasattr(model, 'module'):
        return model.module
    return model

# 사용 시
def _log_gradient_norms(self, runner: Runner) -> None:
    model = self._get_model(runner)  # DDP unwrap
    for name, param in model.named_parameters():
        ...
```

---

### 4. MMEngine 호환성

#### ⚠️ 주의 사항

**a) `after_val_epoch` 메트릭 전달 방식** (Line 766-771)
```python
def after_val_epoch(self, runner: Runner,
                    metrics: Optional[Dict[str, float]] = None) -> None:
```
- MMEngine 1.x에서 `after_val_epoch`는 `metrics` 파라미터를 받지 않음
- Metrics는 `runner.message_hub`를 통해 얻어야 함

**b) Hook priority 미설정**
- 다른 Hook과의 실행 순서가 정의되지 않음
- Gradient 계산 후에 실행되어야 하는데 보장되지 않음

#### ✅ 해결 방안
```python
# MMEngine 1.x 방식으로 metrics 가져오기
def after_val_epoch(self, runner: Runner) -> None:
    """Called after validation epoch."""
    if self.log_val_metrics:
        # Get metrics from message hub
        metrics = runner.message_hub.get_info('val')
        if metrics:
            self._log_validation_metrics(runner, metrics)

# Priority 설정
@HOOKS.register_module()
class EnhancedTensorBoardHook(Hook):
    priority = 'VERY_LOW'  # 다른 hook 후에 실행
```

---

### 5. Image Denormalization

#### ✅ 현재 구현은 대체로 정확함
```python
def _denormalize_image(self, img_tensor: torch.Tensor, mean: tuple, std: tuple):
    img = img_tensor.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)  # x = x * std + mean
    img = torch.clamp(img, 0, 1)
```

#### ⚠️ 개선 가능한 점

**a) Mean/Std 단위 혼동 가능성** (Line 363-371)
```python
mean = (123.675, 116.28, 103.53)  # 0-255 범위
std = (58.395, 57.12, 57.375)     # 0-255 범위

# 0-1 범위로 변환
mean_norm = tuple(m / 255.0 for m in mean)
std_norm = tuple(s / 255.0 for s in std)
```
- Data preprocessor의 mean/std가 이미 0-1 범위일 수 있음
- 자동 감지 로직 필요

**b) Vectorized 연산으로 최적화 가능**
```python
def _denormalize_image(self, img_tensor: torch.Tensor, mean: tuple, std: tuple):
    """Denormalize image tensor to numpy array."""
    # Vectorized operation (더 빠름)
    mean_tensor = torch.tensor(mean, device=img_tensor.device).view(3, 1, 1)
    std_tensor = torch.tensor(std, device=img_tensor.device).view(3, 1, 1)

    img = img_tensor * std_tensor + mean_tensor
    img = torch.clamp(img, 0, 1)

    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    return img_np
```

---

## 추가 발견 사항

### 6. 중복 코드
- `_init_tensorboard_writer`와 `_get_tensorboard_writer`가 거의 동일 (Line 98-162)
- `_init_tensorboard_writer`는 사용되지 않음

### 7. Error Handling 문제
```python
except Exception as e:
    import traceback
    traceback.print_exc()
    pass  # 에러를 삼킴
```
- 에러를 print만 하고 무시함
- 디버깅 어려움
- 최소한 로깅해야 함

### 8. Validation image 매 iteration 로깅
- `val_image_interval=1`이 default
- Validation은 보통 수백~수천 iteration
- 디스크 I/O 폭발

---

## 우선순위별 수정 권장사항

### 🔴 Critical (즉시 수정 필요)
1. DDP wrapper 처리 추가
2. Learning rate interval 체크 추가
3. MMEngine `after_val_epoch` 시그니처 수정

### 🟡 Important (개선 필요)
4. First batch data 메모리 사용량 감소
5. Gradient norm 계산 최적화
6. Validation image interval default 값 증가 (1 -> 10)

### 🟢 Nice to have (선택적 개선)
7. Image denormalization vectorization
8. 중복 코드 제거
9. Error logging 개선
10. Mean/Std 자동 감지

---

## 성능 영향 분석

| 기능 | 현재 오버헤드 | 개선 후 오버헤드 |
|------|--------------|------------------|
| Learning rate logging | 매 iter (~0.1ms) | 100 iter마다 (~0.001ms/iter) |
| Gradient norms | ~5-10ms (GPU 동기화) | ~1-2ms (최적화 후) |
| Training images | ~100-500ms (재추론) | ~50-100ms (outputs 재사용) |
| Weight histograms | ~50-100ms | ~30-50ms (샘플링) |
| First batch storage | 수백 MB RAM | ~10-50 MB (4개만 저장) |

---

## 테스트 권장사항

1. **DDP 환경 테스트**
   ```bash
   torchrun --nproc_per_node=2 train.py
   ```

2. **메모리 프로파일링**
   ```python
   import torch.cuda
   torch.cuda.memory_summary()
   ```

3. **성능 벤치마크**
   - Hook 활성화 전/후 iteration time 비교
   - TensorBoard 파일 크기 모니터링
