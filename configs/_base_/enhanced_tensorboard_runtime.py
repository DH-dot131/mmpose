# Enhanced TensorBoard Runtime Configuration
# 이 파일은 EnhancedTensorBoardHook을 사용하는 예시 설정입니다.
# 기본 default_runtime.py를 상속받아 EnhancedTensorBoardHook을 추가합니다.

_base_ = ['./default_runtime.py']

# custom hooks에 EnhancedTensorBoardHook 추가
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type='SyncBuffersHook'),
    
    # Enhanced TensorBoard Hook - 다양한 학습 정보를 TensorBoard에 기록
    dict(
        type='EnhancedTensorBoardHook',
        # 모든 기능 활성화
        log_grad_norm=True,          # Gradient norm 로깅
        log_weight_hist=True,        # Weight histogram 로깅
        log_val_metrics=True,        # Validation metrics 로깅
        log_train_images=True,       # Training image 로깅
        # log_val_images removed (not effective, causes large file sizes)
        log_lr=True,                 # Learning rate 로깅
        log_model_stats=True,        # Model statistics 로깅
        
        # 로깅 주기 설정
        grad_norm_interval=100,      # 매 100 iteration마다 gradient norm 기록
        weight_hist_interval=1,      # 매 epoch마다 weight histogram 기록
        train_image_interval=500,    # 매 500 iteration마다 training image 기록
        val_image_interval=1,        # 매 validation iteration마다 validation image 기록
        
        # 기타 설정
        max_layers_to_log=20,        # 개별 레이어 로깅 최대 개수
        kpt_thr=0.3,                 # Keypoint 시각화 threshold
    ),
]

# TensorBoard backend는 default_runtime.py에서 이미 설정되어 있음
# 필요시 아래와 같이 추가 설정 가능:
# vis_backends = [
#     dict(type='LocalVisBackend'),
#     dict(type='TensorboardVisBackend'),
# ]

