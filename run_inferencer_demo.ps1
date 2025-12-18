# mmpose inferencer_demo.py 실행 스크립트 (PowerShell)

# 한글 인코딩 설정
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 입력 디렉토리
$INPUT_DIR = "L:\research\project\foot_ap_data_inspection\HVAngleEst\V5\HVAngleEst\images_clahe"
# $INPUT_DIR = "L:\research\project\foot_ap_data_inspection\data\foot_ap_mmpose\images"
# 출력 디렉토리
$OUTPUT_DIR = "L:\research\project\foot_ap_data_inspection\HVAngleEst\V5\HVAngleEst\images_clahe_predict"
# $OUTPUT_DIR = "data\Foot_AP_L\not_labeled/images_predict_M2"
# pose2d 설정 파일
$POSE2D_CONFIG = "L:\research\project\foot_ap_data_inspection\mmpose\configs\body_2d_keypoint\custom\td-hm-hrnet-w32-adam-lr1e-2-warm100batch-bs8-ep100-coco-384x288_AP-base256.py"
# $POSE2D_CONFIG = "L:\research\project\foot_ap_data_inspection\work_dirs\foot_ap\train\hrnet_custom_24_384x288.py"
# pose2d 가중치 파일
$POSE2D_WEIGHTS = "L:\research\project\foot_ap_data_inspection\work_dirs\foot_ap\td-hm-hrnet-w32-adam-lr1e-2-warm100batch-bs8-ep100-coco-384x288_AP-base256_20251215\best_EPE_epoch_72.pth"
# $POSE2D_WEIGHTS = "L:\research\project\foot_ap_data_inspection\work_dirs\foot_ap\train\best_NME_epoch_90.pth"
# detection 모델 설정 파일
$DET_MODEL = "L:\research\project\foot_ap_data_inspection\work_dirs\yolox_s_foot_xray_less_aug\yolox_s_foot_xray_less_aug.py"
# detection 가중치 파일
$DET_WEIGHTS = "L:\research\project\foot_ap_data_inspection\work_dirs\yolox_s_foot_xray_less_aug\epoch_90.pth"

# 출력 디렉토리 생성
if (!(Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR -Force
    Write-Host "Output directory created: $OUTPUT_DIR"
}

# mmpose inferencer_demo.py 실행
Write-Host "Starting inference..."
python demo/inferencer_demo.py `
    $INPUT_DIR `
    --pose2d $POSE2D_CONFIG `
    --pose2d-weights $POSE2D_WEIGHTS `
    --det-model $DET_MODEL `
    --det-weights $DET_WEIGHTS `
    --det-cat-ids 0 1 `
    --draw-bbox `
    --vis-out-dir $OUTPUT_DIR `
    --pred-out-dir $OUTPUT_DIR `
    --show-kpt-idx

Write-Host "Inference completed! Results saved to: $OUTPUT_DIR"
