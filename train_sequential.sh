#!/bin/bash
# mmpose Sequential/Parallel Training Script (Bash)
# Automatically decides sequential or parallel execution based on GPU/CPU resources

# UTF-8 encoding settings
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# ============================================
# Configuration Area - Modify here
# ============================================

# List of config files to train (executed in order)
CONFIG_FILES=(
    "configs/body_2d_keypoint/custom/td-hm_ViTPose-base_1xb8-200e_foot-ap-384x288_2x.py"
    "configs/body_2d_keypoint/custom/td-hm_ViTPose-base_1xb8-200e_foot-ap-384x288_4x.py"
    # Add more as needed
)

# Additional training options (modify if needed)
TRAIN_OPTIONS=(
    # "--amp"  # Enable automatic mixed precision training
    # "--resume"  # Auto resume from latest checkpoint
    # "--no-validate"  # Disable validation (not recommended)
)

# Resource thresholds (adjust if needed)
GPU_MEMORY_THRESHOLD_MB=2000  # Minimum free GPU memory (MB) to allow parallel execution
GPU_UTIL_THRESHOLD=80  # Maximum GPU utilization (%) to allow parallel execution
CPU_CORE_THRESHOLD=4  # Minimum free CPU cores to allow parallel execution
CPU_LOAD_THRESHOLD=80  # Maximum CPU load (%) to allow parallel execution

# Execution mode: "auto", "sequential", or "parallel"
EXECUTION_MODE="auto"

# Show training output in real-time (true/false)
# If true, training output will be displayed on screen while running
SHOW_OUTPUT=true

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
    echo "$log_message" >&2  # Output to stderr to avoid interfering with function return values
    echo "$log_message" >> "$LOG_FILE"
}

# Get GPU information
get_gpu_info() {
    if ! command -v nvidia-smi &> /dev/null; then
        write_log "nvidia-smi not found. GPU monitoring disabled." "WARNING"
        return 1
    fi
    
    # Get total GPU memory (MB)
    local total_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    
    # Get used GPU memory (MB)
    local used_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
    
    # Get free GPU memory (MB)
    local free_mem=$((total_mem - used_mem))
    
    # Get GPU utilization (%)
    local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1)
    
    # Get number of GPUs
    local num_gpus=$(nvidia-smi --list-gpus | wc -l)
    
    echo "$free_mem $gpu_util $num_gpus $total_mem"
}

# Get CPU information
get_cpu_info() {
    # Get total CPU cores
    local total_cores=$(nproc)
    
    # Get CPU load average (1 minute)
    local load_1min=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
    
    # Calculate CPU load percentage
    local cpu_load=$(echo "$load_1min * 100 / $total_cores" | bc -l | awk '{printf "%.0f", $1}')
    
    # Estimate free cores (approximate)
    local free_cores=$(echo "$total_cores * (100 - $cpu_load) / 100" | bc -l | awk '{printf "%.0f", $1}')
    
    echo "$free_cores $cpu_load $total_cores"
}

# Parse config file to extract resource requirements
parse_config() {
    local config_file="$1"
    
    if [ ! -f "$config_file" ]; then
        echo "0 0 0 0"
        return 1
    fi
    
    # Extract batch_size from train_dataloader
    local batch_size=$(grep -A 20 "train_dataloader" "$config_file" | grep "batch_size" | head -n1 | sed -E 's/.*batch_size[[:space:]]*=[[:space:]]*([0-9]+).*/\1/' | head -n1)
    if [ -z "$batch_size" ]; then
        batch_size=8  # default
    fi
    
    # Extract num_workers from train_dataloader
    local num_workers=$(grep -A 20 "train_dataloader" "$config_file" | grep "num_workers" | head -n1 | sed -E 's/.*num_workers[[:space:]]*=[[:space:]]*([0-9]+).*/\1/' | head -n1)
    if [ -z "$num_workers" ]; then
        num_workers=8  # default
    fi
    
    # Extract input_size from codec (approximate image size for memory estimation)
    local input_size=$(grep -A 5 "codec.*=" "$config_file" | grep "input_size" | head -n1 | sed -E 's/.*input_size[[:space:]]*=[[:space:]]*\(([0-9]+),.*/\1/' | head -n1)
    if [ -z "$input_size" ]; then
        input_size=384  # default
    fi
    
    # Estimate GPU memory requirement (rough estimation in MB)
    # Formula: batch_size * input_size_factor * model_factor
    local input_factor=$(echo "$input_size / 256" | bc -l | awk '{printf "%.2f", $1}')
    local estimated_mem=$(echo "$batch_size * $input_factor * 500" | bc -l | awk '{printf "%.0f", $1}')
    
    echo "$batch_size $num_workers $estimated_mem $input_size"
}

# Decide execution mode based on resources
decide_execution_mode() {
    local gpu_info=$(get_gpu_info)
    if [ $? -ne 0 ]; then
        write_log "GPU monitoring unavailable. Using sequential mode." "WARNING"
        echo "sequential"
        return
    fi
    
    local gpu_free_mem=$(echo "$gpu_info" | awk '{print $1}')
    local gpu_util=$(echo "$gpu_info" | awk '{print $2}')
    local num_gpus=$(echo "$gpu_info" | awk '{print $3}')
    local gpu_total_mem=$(echo "$gpu_info" | awk '{print $4}')
    
    local cpu_info=$(get_cpu_info)
    local cpu_free_cores=$(echo "$cpu_info" | awk '{print $1}')
    local cpu_load=$(echo "$cpu_info" | awk '{print $2}')
    local cpu_total_cores=$(echo "$cpu_info" | awk '{print $3}')
    
    write_log "=== System Resource Status ===" "INFO"
    write_log "GPU: ${gpu_free_mem}MB free / ${gpu_total_mem}MB total, ${gpu_util}% utilized" "INFO"
    write_log "CPU: ${cpu_free_cores} cores free / ${cpu_total_cores} total, ${cpu_load}% load" "INFO"
    write_log "Number of GPUs: ${num_gpus}" "INFO"
    
    # Analyze all config files
    local total_estimated_mem=0
    local total_num_workers=0
    local max_batch_size=0
    
    write_log "" "INFO"
    write_log "=== Config File Analysis ===" "INFO"
    
    for config_file in "${CONFIG_FILES[@]}"; do
        local config_info=$(parse_config "$config_file")
        local batch_size=$(echo "$config_info" | awk '{print $1}')
        local num_workers=$(echo "$config_info" | awk '{print $2}')
        local estimated_mem=$(echo "$config_info" | awk '{print $3}')
        local input_size=$(echo "$config_info" | awk '{print $4}')
        
        write_log "  ${config_file}:" "INFO"
        write_log "    - batch_size: ${batch_size}, num_workers: ${num_workers}" "INFO"
        write_log "    - estimated GPU memory: ~${estimated_mem}MB, input_size: ${input_size}" "INFO"
        
        total_estimated_mem=$((total_estimated_mem + estimated_mem))
        total_num_workers=$((total_num_workers + num_workers))
        if [ $batch_size -gt $max_batch_size ]; then
            max_batch_size=$batch_size
        fi
    done
    
    write_log "" "INFO"
    write_log "Total estimated GPU memory for all jobs: ~${total_estimated_mem}MB" "INFO"
    write_log "Total num_workers: ${total_num_workers}" "INFO"
    
    # Decision logic
    if [ "$EXECUTION_MODE" = "sequential" ]; then
        write_log "Execution mode: SEQUENTIAL (forced by user)" "INFO"
        echo "sequential"
        return
    elif [ "$EXECUTION_MODE" = "parallel" ]; then
        write_log "Execution mode: PARALLEL (forced by user)" "INFO"
        echo "parallel"
        return
    fi
    
    # Auto decision
    local can_parallel=true
    
    # Check GPU memory
    if [ $gpu_free_mem -lt $((total_estimated_mem + GPU_MEMORY_THRESHOLD_MB)) ]; then
        write_log "Insufficient GPU memory for parallel execution" "INFO"
        can_parallel=false
    fi
    
    # Check GPU utilization
    if [ $gpu_util -gt $GPU_UTIL_THRESHOLD ]; then
        write_log "GPU utilization too high (${gpu_util}% > ${GPU_UTIL_THRESHOLD}%)" "INFO"
        can_parallel=false
    fi
    
    # Check CPU cores
    if [ $cpu_free_cores -lt $((total_num_workers + CPU_CORE_THRESHOLD)) ]; then
        write_log "Insufficient CPU cores for parallel execution" "INFO"
        can_parallel=false
    fi
    
    # Check CPU load
    if [ $cpu_load -gt $CPU_LOAD_THRESHOLD ]; then
        write_log "CPU load too high (${cpu_load}% > ${CPU_LOAD_THRESHOLD}%)" "INFO"
        can_parallel=false
    fi
    
    # Check if only one GPU available
    if [ $num_gpus -eq 1 ] && [ ${#CONFIG_FILES[@]} -gt 1 ]; then
        write_log "Only one GPU available. Parallel execution may cause memory issues." "INFO"
        # Still allow if memory is sufficient
        if [ $gpu_free_mem -lt $((total_estimated_mem * 2)) ]; then
            can_parallel=false
        fi
    fi
    
    if [ "$can_parallel" = true ]; then
        write_log "Execution mode: PARALLEL (auto-selected)" "INFO"
        echo "parallel"
    else
        write_log "Execution mode: SEQUENTIAL (auto-selected)" "INFO"
        echo "sequential"
    fi
}

# Training execution function
start_training() {
    local config_file="$1"
    local index="$2"
    local total="$3"
    local is_background="${4:-false}"
    
    write_log "========================================" "INFO"
    write_log "Training started: [${index}/${total}] ${config_file}" "INFO"
    if [ "$is_background" = "true" ]; then
        write_log "Running in background" "INFO"
    fi
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
    
    # Create separate log file for each training job
    local job_log_file="${LOG_DIR}/train_job_${index}_${TIMESTAMP}.log"
    
    write_log "Command: ${train_cmd}" "INFO"
    write_log "Job log: ${job_log_file}" "INFO"
    write_log "----------------------------------------" "INFO"
    
    # Execute training
    local start_time=$(date +%s)
    
    if [ "$is_background" = "true" ]; then
        # Background execution (parallel mode)
        # Output is saved to log file, can be viewed with: tail -f "$job_log_file"
        eval "$train_cmd" >> "$job_log_file" 2>&1 &
        local pid=$!
        if [ "$SHOW_OUTPUT" = "true" ]; then
            write_log "Background job started (PID: ${pid})" "INFO"
            write_log "View output with: tail -f ${job_log_file}" "INFO"
        fi
        echo "$pid"
        return 0
    else
        # Foreground execution
        if [ "$SHOW_OUTPUT" = "true" ]; then
            # Show output in real-time using tee
            write_log "Training output:" "INFO"
            write_log "----------------------------------------" "INFO"
            if eval "$train_cmd" 2>&1 | tee "$job_log_file"; then
                local exit_code=${PIPESTATUS[0]}
            else
                local exit_code=$?
            fi
        else
            # Silent execution (only log file)
            if eval "$train_cmd" >> "$job_log_file" 2>&1; then
                local exit_code=0
            else
                local exit_code=$?
            fi
        fi
        
        if [ $exit_code -eq 0 ]; then
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
            write_log "----------------------------------------" "ERROR"
            write_log "Error occurred: [${index}/${total}] ${config_file}" "ERROR"
            write_log "Error message: Training command failed with exit code ${exit_code}" "ERROR"
            write_log "========================================" "ERROR"
            return 1
        fi
    fi
}

# Wait for background jobs
wait_for_jobs() {
    local pids=("$@")
    local failed_pids=()
    
    write_log "Waiting for ${#pids[@]} background jobs to complete..." "INFO"
    
    for pid in "${pids[@]}"; do
        if wait $pid; then
            write_log "Job PID ${pid} completed successfully" "INFO"
        else
            write_log "Job PID ${pid} failed with exit code $?" "ERROR"
            failed_pids+=("$pid")
        fi
    done
    
    if [ ${#failed_pids[@]} -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

# ============================================
# Main Execution
# ============================================

write_log "Sequential/Parallel training script started" "INFO"
write_log "Total training jobs: ${#CONFIG_FILES[@]}" "INFO"
write_log "Log file: ${LOG_FILE}" "INFO"
write_log ""

# Decide execution mode
EXEC_MODE=$(decide_execution_mode)
write_log "" "INFO"
write_log "=== Starting Training ===" "INFO"
write_log "" "INFO"

success_count=0
fail_count=0
failed_configs=()
background_pids=()

if [ "$EXEC_MODE" = "parallel" ]; then
    # Parallel execution
    write_log "Starting all jobs in parallel..." "INFO"
    
    for i in "${!CONFIG_FILES[@]}"; do
        config_file="${CONFIG_FILES[$i]}"
        index=$((i + 1))
        total=${#CONFIG_FILES[@]}
        
        pid=$(start_training "$config_file" "$index" "$total" "true")
        if [ $? -eq 0 ]; then
            background_pids+=("$pid")
            write_log "Started job [${index}/${total}] with PID ${pid}" "INFO"
        else
            ((fail_count++))
            failed_configs+=("$config_file")
        fi
        
        # Small delay between starting jobs
        sleep 2
    done
    
    # Wait for all jobs to complete
    if wait_for_jobs "${background_pids[@]}"; then
        success_count=${#background_pids[@]}
    else
        fail_count=$((fail_count + ${#failed_configs[@]}))
    fi
    
else
    # Sequential execution
    for i in "${!CONFIG_FILES[@]}"; do
        config_file="${CONFIG_FILES[$i]}"
        index=$((i + 1))
        total=${#CONFIG_FILES[@]}
        
        if start_training "$config_file" "$index" "$total" "false"; then
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
fi

# ============================================
# Final Summary
# ============================================

write_log ""
write_log "========================================" "INFO"
write_log "Training completed" "INFO"
write_log "========================================" "INFO"
write_log "Execution mode: ${EXEC_MODE}" "INFO"
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
write_log "Main log file: ${LOG_FILE}" "INFO"
if [ "$EXEC_MODE" = "parallel" ]; then
    write_log "Individual job logs: ${LOG_DIR}/train_job_*_${TIMESTAMP}.log" "INFO"
fi
write_log "========================================" "INFO"

# Return exit code 1 if any job failed
if [ $fail_count -gt 0 ]; then
    exit 1
else
    exit 0
fi
