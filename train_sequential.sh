#!/bin/bash
# mmpose Sequential Training Script (Bash)
# Template for training multiple config files sequentially

# UTF-8 encoding settings
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# ============================================
# Configuration Area - Modify here
# ============================================

# List of config files to train (executed in order)
CONFIG_FILES=(
    "configs/body_2d_keypoint/custom/td-hm-hrnet-w48-adam-lr1e-2-warm100batch-bs8-ep100-coco-384x288_AP-base256.py"
    "configs/body_2d_keypoint/custom/td-hm_ViTPose-base_1xb8-100e_foot-ap-384x288.py"
    "configs/body_2d_keypoint/custom/td-hm_ViTPose-base_1xb16-100e_foot-ap-256x192.py"
    # Add more as needed
)

# Additional training options (modify if needed)
TRAIN_OPTIONS=(
    # "--amp"  # Enable automatic mixed precision training
    # "--resume"  # Auto resume from latest checkpoint
    # "--no-validate"  # Disable validation (not recommended)
)

# Log file directory
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_sequential_${TIMESTAMP}.log"

# ============================================
# Script Start
# ============================================

# Set current directory to mmpose directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create log directory
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# Logging function
write_log() {
    local message="$1"
    local level="${2:-INFO}"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    local log_message="[${timestamp}] [${level}] ${message}"
    echo "$log_message"
    echo "$log_message" >> "$LOG_FILE"
}

# Training execution function
start_training() {
    local config_file="$1"
    local index="$2"
    local total="$3"
    
    write_log "========================================" "INFO"
    write_log "Training started: [${index}/${total}] ${config_file}" "INFO"
    write_log "========================================" "INFO"
    
    # Check if config file exists
    if [ ! -f "$config_file" ]; then
        write_log "Error: Config file not found: ${config_file}" "ERROR"
        return 1
    fi
    
    # Build training command
    local train_cmd="python -X utf8 tools/train.py \"${config_file}\""
    
    if [ ${#TRAIN_OPTIONS[@]} -gt 0 ]; then
        train_cmd="${train_cmd} ${TRAIN_OPTIONS[*]}"
    fi
    
    write_log "Command: ${train_cmd}" "INFO"
    write_log "----------------------------------------" "INFO"
    
    # Execute training
    local start_time=$(date +%s)
    
    if eval "$train_cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        local seconds=$((duration % 60))
        local duration_str=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)
        
        write_log "----------------------------------------" "INFO"
        write_log "Training completed: [${index}/${total}] ${config_file}" "INFO"
        write_log "Duration: ${duration_str}" "INFO"
        write_log "========================================" "INFO"
        
        return 0
    else
        write_log "Error occurred: [${index}/${total}] ${config_file}" "ERROR"
        write_log "Error message: Training command failed with exit code $?" "ERROR"
        write_log "========================================" "ERROR"
        return 1
    fi
}

# ============================================
# Main Execution
# ============================================

write_log "Sequential training script started" "INFO"
write_log "Total training jobs: ${#CONFIG_FILES[@]}" "INFO"
write_log "Log file: ${LOG_FILE}" "INFO"
write_log ""

success_count=0
fail_count=0
failed_configs=()

for i in "${!CONFIG_FILES[@]}"; do
    config_file="${CONFIG_FILES[$i]}"
    index=$((i + 1))
    total=${#CONFIG_FILES[@]}
    
    if start_training "$config_file" "$index" "$total"; then
        ((success_count++))
    else
        ((fail_count++))
        failed_configs+=("$config_file")
    fi
    
    # Wait before next training if not the last job
    if [ $i -lt $((${#CONFIG_FILES[@]} - 1)) ]; then
        write_log "Waiting 10 seconds before next training..." "INFO"
        sleep 10
    fi
done

# ============================================
# Final Summary
# ============================================

write_log ""
write_log "========================================" "INFO"
write_log "Sequential training completed" "INFO"
write_log "========================================" "INFO"
write_log "Success: ${success_count} / ${#CONFIG_FILES[@]}" "INFO"
write_log "Failed: ${fail_count} / ${#CONFIG_FILES[@]}" "INFO"

if [ ${#failed_configs[@]} -gt 0 ]; then
    write_log ""
    write_log "Failed config files:" "WARNING"
    for failed_config in "${failed_configs[@]}"; do
        write_log "  - ${failed_config}" "WARNING"
    done
fi

write_log ""
write_log "Log file: ${LOG_FILE}" "INFO"
write_log "========================================" "INFO"

# Return exit code 1 if any job failed
if [ $fail_count -gt 0 ]; then
    exit 1
else
    exit 0
fi

