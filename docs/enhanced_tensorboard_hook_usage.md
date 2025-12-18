# Enhanced TensorBoard Hook 사용 가이드

## 개요

`EnhancedTensorBoardHook`은 mmpose 학습 과정에서 TensorBoard에 다양한 정보를 기록하는 hook입니다. 기본 `LoggerHook`이 제공하는 loss, learning rate, memory 정보 외에도 다음과 같은 추가 정보를 제공합니다:

- **Gradient Norms**: 레이어별 및 전체 평균 gradient norm
- **Weight Histograms**: 레이어별 가중치 분포
- **Validation Metrics**: PCK, NME, AUC, EPE 등 모든 검증 메트릭
- **Training/Validation Images**: 예측 결과와 ground truth를 포함한 이미지
- **Learning Rate Schedule**: 학습률 변화 추적
- **Model Statistics**: 모델 파라미터 개수 및 크기

## 설정 방법

### 1. Config 파일에 Hook 추가

config 파일의 `custom_hooks` 섹션에 `EnhancedTensorBoardHook`을 추가합니다:

```python
# config 파일 예시
_base_ = ['./default_runtime.py']

# ... 다른 설정들 ...

# custom hooks
custom_hooks = [
    dict(type='SyncBuffersHook'),  # 기존 hook
    dict(
        type='EnhancedTensorBoardHook',
        # 모든 기능 활성화 (기본값)
        log_grad_norm=True,
        log_weight_hist=True,
        log_val_metrics=True,
        log_train_images=True,
        log_val_images=True,
        log_lr=True,
        log_model_stats=True,
        # 로깅 주기 설정
        grad_norm_interval=100,      # 매 100 iteration마다 gradient norm 기록
        weight_hist_interval=1,      # 매 epoch마다 weight histogram 기록
        train_image_interval=500,    # 매 500 iteration마다 training image 기록
        val_image_interval=1,        # 매 validation iteration마다 validation image 기록
        max_layers_to_log=20,        # 개별 레이어 로깅 최대 개수
        kpt_thr=0.3,                 # keypoint 시각화 threshold
    ),
]
```

### 2. TensorBoard Backend 활성화

config 파일에서 `TensorboardVisBackend`가 활성화되어 있는지 확인합니다:

```python
# visualizer 설정
vis_backends = [
    dict(type='LocalVisBackend'),      # 로컬 저장 (선택사항)
    dict(type='TensorboardVisBackend'), # TensorBoard (필수)
]

visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
```

## 파라미터 설명

### 주요 파라미터

- `log_grad_norm` (bool): Gradient norm 로깅 활성화 여부. Default: True
- `log_weight_hist` (bool): Weight histogram 로깅 활성화 여부. Default: True
- `log_val_metrics` (bool): Validation metrics 로깅 활성화 여부. Default: True
- `log_train_images` (bool): Training image 로깅 활성화 여부. Default: True
- `log_val_images` (bool): Validation image 로깅 활성화 여부. Default: True
- `log_lr` (bool): Learning rate 로깅 활성화 여부. Default: True
- `log_model_stats` (bool): Model statistics 로깅 활성화 여부. Default: True

### 로깅 주기 파라미터

- `grad_norm_interval` (int): Gradient norm 로깅 주기 (iteration 단위). Default: 100
- `weight_hist_interval` (int): Weight histogram 로깅 주기 (epoch 단위). Default: 1
- `train_image_interval` (int): Training image 로깅 주기 (iteration 단위). Default: 500
- `val_image_interval` (int): Validation image 로깅 주기 (iteration 단위). Default: 1
- `max_layers_to_log` (int): 개별 레이어 로깅 최대 개수. Default: 20

### 기타 파라미터

- `kpt_thr` (float): Keypoint 시각화 threshold. Default: 0.3
- `backend_args` (dict): File I/O backend 인자. Default: None

## 사용 예시

### 기본 사용 (모든 기능 활성화)

```python
custom_hooks = [
    dict(type='EnhancedTensorBoardHook'),
]
```

### 선택적 기능 활성화

```python
custom_hooks = [
    dict(
        type='EnhancedTensorBoardHook',
        log_train_images=False,  # Training image 로깅 비활성화
        log_val_images=False,     # Validation image 로깅 비활성화
        grad_norm_interval=200,   # Gradient norm 로깅 주기 조정
    ),
]
```

### 성능 최적화 (빈도 감소)

```python
custom_hooks = [
    dict(
        type='EnhancedTensorBoardHook',
        grad_norm_interval=500,      # Gradient norm 로깅 빈도 감소
        train_image_interval=1000,   # Training image 로깅 빈도 감소
        weight_hist_interval=5,     # Weight histogram을 5 epoch마다 기록
        max_layers_to_log=10,       # 로깅할 레이어 수 감소
    ),
]
```

## TensorBoard에서 확인하기

### 1. TensorBoard 실행

학습이 시작되면 다음 명령어로 TensorBoard를 실행합니다:

```bash
tensorboard --logdir=work_dirs
```

또는 특정 실험 디렉토리를 지정:

```bash
tensorboard --logdir=work_dirs/your_experiment_name
```

### 2. TensorBoard에서 확인할 수 있는 정보

#### SCALARS 탭
- `train/gradient_norm/global`: 전체 평균 gradient norm
- `train/gradient_norm/layers/{layer_name}`: 레이어별 gradient norm
- `train/learning_rate/{group_name}`: Learning rate
- `val/{metric_name}`: Validation metrics (PCK, NME, AUC, EPE 등)
- `model/total_parameters`: 전체 파라미터 개수
- `model/trainable_parameters`: 학습 가능한 파라미터 개수
- `model/model_size_mb`: 모델 크기 (MB)

#### HISTOGRAMS 탭
- `weights/{layer_name}`: 레이어별 가중치 분포

#### IMAGES 탭
- `train/image`: Training 이미지 (예측 결과 + ground truth)
- `val/image`: Validation 이미지 (예측 결과 + ground truth)

## 주의사항

1. **성능 영향**: 이미지 로깅은 디스크 I/O와 메모리를 사용하므로, 학습 속도에 영향을 줄 수 있습니다. 필요에 따라 `train_image_interval`과 `val_image_interval`을 조정하세요.

2. **디스크 공간**: 이미지와 histogram 로깅은 많은 디스크 공간을 사용할 수 있습니다. 특히 `val_image_interval=1`로 설정하면 validation 이미지가 매우 많이 기록됩니다.

3. **레이어 수 제한**: `max_layers_to_log` 파라미터로 로깅할 레이어 수를 제한할 수 있습니다. 모델이 큰 경우 이 값을 조정하여 TensorBoard 로딩 속도를 개선할 수 있습니다.

4. **TensorBoard Backend 필수**: `TensorboardVisBackend`가 config에 포함되어 있어야 합니다. 그렇지 않으면 hook이 작동하지 않습니다.

## 문제 해결

### TensorBoard에 정보가 표시되지 않는 경우

1. `TensorboardVisBackend`가 config에 포함되어 있는지 확인
2. `work_dir`에 TensorBoard event 파일이 생성되는지 확인
3. Hook이 `custom_hooks`에 올바르게 추가되었는지 확인

### 메모리 부족 오류

1. `train_image_interval`과 `val_image_interval`을 증가시켜 이미지 로깅 빈도 감소
2. `max_layers_to_log` 값을 감소시켜 로깅할 레이어 수 제한
3. `log_train_images=False` 또는 `log_val_images=False`로 설정하여 이미지 로깅 비활성화

## 참고

- 기본 `LoggerHook`은 여전히 loss, learning rate, memory 등의 기본 정보를 기록합니다.
- `EnhancedTensorBoardHook`은 추가 정보를 제공하며, 기본 hook을 대체하지 않습니다.
- 두 hook을 함께 사용하는 것이 권장됩니다.

