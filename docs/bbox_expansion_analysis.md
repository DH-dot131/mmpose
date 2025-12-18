# mmpose inferencer_demo.py의 BBox Expansion 분석

## 조사 결과

**결론: `inferencer_demo.py`를 통해 예측할 때 bbox expansion이 일어납니다.**

## 상세 분석

### 1. 처리 흐름

1. **inferencer_demo.py** → `MMPoseInferencer` 초기화
2. **Pose2DInferencer.preprocess_single()** → Detection 모델로부터 bbox 획득
3. **Pipeline 적용** → `self.pipeline(inst)` 호출
4. **GetBBoxCenterScale Transform** → bbox를 center, scale로 변환하며 expansion 적용

### 2. BBox Expansion 메커니즘

#### GetBBoxCenterScale Transform
- **위치**: `mmpose/mmpose/datasets/transforms/common_transforms.py`
- **기본 padding 값**: `1.25`
- **코드**:
```python
def __init__(self, padding: float = 1.25) -> None:
    self.padding = padding

def transform(self, results: Dict) -> Optional[dict]:
    bbox = results['bbox']
    center, scale = bbox_xyxy2cs(bbox, padding=self.padding)
    results['bbox_center'] = center
    results['bbox_scale'] = scale
```

#### bbox_xyxy2cs 함수
- **위치**: `mmpose/mmpose/structures/bbox/transforms.py`
- **Expansion 로직**:
```python
def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    # bbox 크기를 padding 배수로 확장
    scale = (bbox[..., 2:] - bbox[..., :2]) * padding
    center = (bbox[..., 2:] + bbox[..., :2]) * 0.5
    return center, scale
```

### 3. Config 파일 확인

현재 사용 중인 config 파일 (`td-hm-hrnet-w32-adam-lr1e-2-warm100batch-bs8-ep100-coco-384x288_AP-base256.py`)에서:

```python
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),  # ← padding 기본값 1.25 사용
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]
```

**`GetBBoxCenterScale`에 padding 파라미터가 명시되지 않았으므로 기본값 1.25가 사용됩니다.**

### 4. Expansion 효과

- **원본 bbox 크기**: `(x2 - x1, y2 - y1)`
- **Expansion 후 scale**: `(x2 - x1) * 1.25, (y2 - y1) * 1.25`
- **실제 확장**: bbox가 **25% 확장**됩니다 (각 방향으로 약 12.5%씩)

### 5. Inferencer에서의 적용

`Pose2DInferencer.preprocess_single()` 메서드에서:

```python
# Detection 모델로부터 bbox 획득
bboxes = detection_results...

# Pipeline 적용 (GetBBoxCenterScale 포함)
for bbox in bboxes:
    inst = data_info.copy()
    inst['bbox'] = bbox[None, :4]
    inst['bbox_score'] = bbox[4:5]
    data_infos.append(self.pipeline(inst))  # ← 여기서 expansion 발생
```

`self.pipeline`은 `cfg.test_dataloader.dataset.pipeline`에서 가져오며, 이 pipeline에는 `GetBBoxCenterScale` transform이 포함되어 있습니다.

## 결론

1. ✅ **bbox expansion이 일어납니다**
2. ✅ **기본 padding 값은 1.25** (25% 확장)
3. ✅ **모든 top-down 모델의 inference에서 적용됩니다**
4. ✅ **Config 파일에서 padding 값을 명시적으로 변경하지 않으면 기본값 1.25가 사용됩니다**

## Padding 값 변경 방법

만약 padding 값을 변경하고 싶다면, config 파일의 pipeline에서 다음과 같이 수정할 수 있습니다:

```python
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.0),  # expansion 없음
    # 또는
    dict(type='GetBBoxCenterScale', padding=1.5),  # 50% 확장
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]
```

## 참고

- `GetBBoxCenterScale` transform은 학습 시에도 동일하게 적용됩니다 (train_pipeline에도 포함)
- Expansion은 키포인트가 bbox 경계 근처에 있을 때 잘리지 않도록 하기 위한 일반적인 기법입니다
- Padding 값 1.25는 일반적으로 사용되는 표준 값입니다

