#!/bin/bash
# mmpose Sequential Testing Script (Bash)
# Automatically finds completed training runs and tests them

# UTF-8 encoding settings
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# ============================================
# Configuration Area - Modify here
# ============================================

# Work directory to search for completed trainings
WORK_DIR_BASE="../work_dirs/foot_ap"

# Target epoch to check completion (default: 100)
TARGET_EPOCH=200

# Checkpoint pattern to search for
CHECKPOINT_PATTERN="best_EPE_epoch_*.pth"

# Dump file name
DUMP_FILE="results_v3.pkl"

# Additional test options (modify if needed)
TEST_OPTIONS=(
    # "--show-dir" "path/to/show"  # Save visualization images
    # "--show"  # Display prediction results
)

# Execution mode: "auto", "sequential", or "parallel"
EXECUTION_MODE="auto"

# Show test output in real-time (true/false)
# If true, test output will be displayed on screen while running
SHOW_OUTPUT=true

# Resource thresholds (for parallel execution)
GPU_MEMORY_THRESHOLD_MB=2000
GPU_UTIL_THRESHOLD=80
CPU_CORE_THRESHOLD=4
CPU_LOAD_THRESHOLD=80

# Log file directory
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/test_sequential_${TIMESTAMP}.log"

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
    
    local total_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    local used_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
    local free_mem=$((total_mem - used_mem))
    local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1)
    local num_gpus=$(nvidia-smi --list-gpus | wc -l)
    
    echo "$free_mem $gpu_util $num_gpus $total_mem"
}

# Get CPU information
get_cpu_info() {
    local total_cores=$(nproc)
    local load_1min=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
    local cpu_load=$(echo "$load_1min * 100 / $total_cores" | bc -l | awk '{printf "%.0f", $1}')
    local free_cores=$(echo "$total_cores * (100 - $cpu_load) / 100" | bc -l | awk '{printf "%.0f", $1}')
    
    echo "$free_cores $cpu_load $total_cores"
}

# Check if training completed up to target epoch
check_training_completed() {
    local work_dir="$1"
    local target_epoch="$2"
    
    # Check for log files that indicate completion
    # Look for log files in the work_dir or timestamp subdirectories
    local log_files=()
    
    # Check in work_dir directly
    if [ -d "$work_dir" ]; then
        # Look for timestamp subdirectories
        for timestamp_dir in "$work_dir"/*/; do
            if [ -d "$timestamp_dir" ]; then
                # Check for log files
                local log_file=$(find "$timestamp_dir" -maxdepth 1 -name "*.log" -type f | head -n1)
                if [ -n "$log_file" ]; then
                    log_files+=("$log_file")
                fi
            fi
        done
        
        # Also check work_dir directly
        local direct_log=$(find "$work_dir" -maxdepth 1 -name "*.log" -type f | head -n1)
        if [ -n "$direct_log" ]; then
            log_files+=("$direct_log")
        fi
    fi
    
    # Check log files for epoch completion
    for log_file in "${log_files[@]}"; do
        if [ -f "$log_file" ]; then
            # Check if log contains target epoch completion
            if grep -q "epoch.*${target_epoch}" "$log_file" 2>/dev/null || \
               grep -q "Epoch\[${target_epoch}\]" "$log_file" 2>/dev/null || \
               grep -q "epoch=${target_epoch}" "$log_file" 2>/dev/null; then
                return 0
            fi
        fi
    done
    
    # Alternative: Check for checkpoint files with epoch >= target_epoch
    # Check for exact epoch match first
    local checkpoint_files=$(find "$work_dir" -name "epoch_${target_epoch}.pth" -o -name "*epoch_${target_epoch}*.pth" 2>/dev/null)
    if [ -n "$checkpoint_files" ]; then
        return 0
    fi
    
    # Check if latest checkpoint epoch >= target_epoch
    # Include both epoch_*.pth and best_EPE_epoch_*.pth files
    local latest_epoch=$(find "$work_dir" -name "epoch_*.pth" -o -name "best_EPE_epoch_*.pth" 2>/dev/null | \
        sed -E 's/.*epoch_([0-9]+)\.pth/\1/' | sort -n | tail -n1)
    if [ -n "$latest_epoch" ] && [ "$latest_epoch" -ge "$target_epoch" ]; then
        return 0
    fi
    
    return 1
}

# Find config file from work_dir
find_config_file() {
    local work_dir="$1"
    
    # Extract config name from work_dir path
    # work_dirs/foot_ap/config_name/timestamp -> config_name
    local config_name=$(basename "$work_dir")
    local parent_dir=$(dirname "$work_dir")
    
    # Priority 1: Check parent directory first (config file is in experiment folder, not in timestamp subdir)
    # If work_dir is a timestamp subdirectory, check parent experiment directory
    if [ -d "$parent_dir" ] && [ "$parent_dir" != "$work_dir" ]; then
        # Check if parent_dir is the experiment folder (not work_dirs/foot_ap itself)
        local parent_basename=$(basename "$parent_dir")
        if [[ ! "$parent_basename" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
            # This is an experiment folder, check for config file
            local parent_config=$(find "$parent_dir" -maxdepth 1 -name "*.py" -type f 2>/dev/null | head -n1)
            if [ -n "$parent_config" ] && [ -f "$parent_config" ]; then
                write_log "  Found config in experiment folder: ${parent_config}" "INFO"
                echo "$parent_config"
                return 0
            fi
        fi
    fi
    
    # Priority 2: Check if config file exists directly in work_dir
    # This is the actual config file used for that specific experiment
    local config_in_workdir=$(find "$work_dir" -maxdepth 1 -name "*.py" -type f 2>/dev/null | head -n1)
    if [ -n "$config_in_workdir" ] && [ -f "$config_in_workdir" ]; then
        write_log "  Found config in work_dir: ${config_in_workdir}" "INFO"
        echo "$config_in_workdir"
        return 0
    fi
    
    # Priority 3: Check timestamp subdirectories for config files (fallback)
    for timestamp_dir in "$work_dir"/*/; do
        if [ -d "$timestamp_dir" ]; then
            local timestamp_config=$(find "$timestamp_dir" -maxdepth 1 -name "*.py" -type f 2>/dev/null | head -n1)
            if [ -n "$timestamp_config" ] && [ -f "$timestamp_config" ]; then
                write_log "  Found config in timestamp dir: ${timestamp_config}" "INFO"
                echo "$timestamp_config"
                return 0
            fi
        fi
    done
    
    # Priority 4: Try to find config file in configs directory by name
    # configs/body_2d_keypoint/custom/${config_name}.py
    # If work_dir is timestamp subdirectory, use parent directory name
    if [[ "$config_name" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
        # This is a timestamp directory, get parent directory name
        config_name=$(basename "$parent_dir")
    fi
    local config_file="configs/body_2d_keypoint/custom/${config_name}.py"
    if [ -f "$config_file" ]; then
        write_log "  Found config in configs directory: ${config_file}" "INFO"
        echo "$config_file"
        return 0
    fi
    
    # Priority 5: Search in all config directories
    local found_config=$(find configs -name "${config_name}.py" -type f 2>/dev/null | head -n1)
    if [ -n "$found_config" ]; then
        write_log "  Found config via search: ${found_config}" "INFO"
        echo "$found_config"
        return 0
    fi
    
    return 1
}

# Find best EPE checkpoint
find_best_epe_checkpoint() {
    local work_dir="$1"
    
    # Search for best_EPE_epoch_*.pth
    local checkpoint=$(find "$work_dir" -name "$CHECKPOINT_PATTERN" -type f 2>/dev/null | head -n1)
    
    if [ -n "$checkpoint" ]; then
        echo "$checkpoint"
        return 0
    fi
    
    return 1
}

# Discover all completed training runs
discover_completed_runs() {
    local work_dir_base="$1"
    local target_epoch="$2"
    local completed_runs=()
    
    if [ ! -d "$work_dir_base" ]; then
        write_log "Work directory not found: ${work_dir_base}" "WARNING"
        return 1
    fi
    
    write_log "Scanning ${work_dir_base} for completed training runs..." "INFO"
    
    # Iterate through all subdirectories in work_dir_base
    for work_dir in "$work_dir_base"/*/; do
        if [ ! -d "$work_dir" ]; then
            continue
        fi
        
        local dir_name=$(basename "$work_dir")
        write_log "  Checking directory: ${dir_name}" "INFO"
        
        # First, check if checkpoint exists in config directory itself
        local checkpoint=$(find_best_epe_checkpoint "$work_dir")
        if [ -n "$checkpoint" ]; then
            # Check if training completed
            if check_training_completed "$work_dir" "$target_epoch"; then
                # Find the most recent timestamp subdirectory to use as work_dir
                local latest_timestamp=$(find "$work_dir" -maxdepth 1 -type d -name "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]" | sort | tail -n1)
                if [ -n "$latest_timestamp" ]; then
                    completed_runs+=("${latest_timestamp}|${checkpoint}")
                    write_log "  Found completed run: ${latest_timestamp} (checkpoint in parent)" "INFO"
                else
                    completed_runs+=("${work_dir}|${checkpoint}")
                    write_log "  Found completed run: ${work_dir}" "INFO"
                fi
                continue
            fi
        fi
        
        # If no checkpoint in config dir, check timestamp subdirectories
        for subdir in "$work_dir"*/; do
            if [ -d "$subdir" ]; then
                local subdir_name=$(basename "$subdir")
                # Check if it looks like a timestamp (YYYYMMDD_HHMMSS format)
                if [[ "$subdir_name" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
                    write_log "    Checking timestamp subdir: ${subdir_name}" "INFO"
                    if check_training_completed "$subdir" "$target_epoch"; then
                        local subdir_checkpoint=$(find_best_epe_checkpoint "$subdir")
                        if [ -n "$subdir_checkpoint" ]; then
                            completed_runs+=("${subdir}|${subdir_checkpoint}")
                            write_log "  Found completed run: ${subdir}" "INFO"
                        fi
                    fi
                fi
            fi
        done
    done
    
    # Output as array (space-separated, will be split later)
    printf '%s\n' "${completed_runs[@]}"
}

# Decide execution mode
decide_execution_mode() {
    local num_jobs="$1"
    
    if [ "$EXECUTION_MODE" = "sequential" ]; then
        echo "sequential"
        return
    elif [ "$EXECUTION_MODE" = "parallel" ]; then
        echo "parallel"
        return
    fi
    
    # Auto decision
    if [ $num_jobs -le 1 ]; then
        echo "sequential"
        return
    fi
    
    local gpu_info=$(get_gpu_info)
    if [ $? -ne 0 ]; then
        echo "sequential"
        return
    fi
    
    local gpu_free_mem=$(echo "$gpu_info" | awk '{print $1}')
    local gpu_util=$(echo "$gpu_info" | awk '{print $2}')
    local cpu_info=$(get_cpu_info)
    local cpu_free_cores=$(echo "$cpu_info" | awk '{print $1}')
    local cpu_load=$(echo "$cpu_info" | awk '{print $2}')
    
    local can_parallel=true
    
    if [ $gpu_free_mem -lt $GPU_MEMORY_THRESHOLD_MB ]; then
        can_parallel=false
    fi
    if [ $gpu_util -gt $GPU_UTIL_THRESHOLD ]; then
        can_parallel=false
    fi
    if [ $cpu_free_cores -lt $CPU_CORE_THRESHOLD ]; then
        can_parallel=false
    fi
    if [ $cpu_load -gt $CPU_LOAD_THRESHOLD ]; then
        can_parallel=false
    fi
    
    if [ "$can_parallel" = true ]; then
        echo "parallel"
    else
        echo "sequential"
    fi
}

# Test execution function
start_testing() {
    local work_dir="$1"
    local checkpoint="$2"
    local config_file="$3"
    local index="$4"
    local total="$5"
    local is_background="${6:-false}"
    
    write_log "========================================" "INFO"
    write_log "Testing started: [${index}/${total}]" "INFO"
    write_log "  Work dir: ${work_dir}" "INFO"
    write_log "  Checkpoint: ${checkpoint}" "INFO"
    write_log "  Config: ${config_file}" "INFO"
    if [ "$is_background" = "true" ]; then
        write_log "  Running in background" "INFO"
    fi
    write_log "========================================" "INFO"
    
    # Check if files exist
    if [ ! -f "$checkpoint" ]; then
        write_log "Error: Checkpoint not found: ${checkpoint}" "ERROR"
        return 1
    fi
    
    if [ ! -f "$config_file" ]; then
        write_log "Error: Config file not found: ${config_file}" "ERROR"
        return 1
    fi
    
    # Create dump file path in experiment folder (parent of timestamp subdir if work_dir is timestamp)
    # Checkpoint files are in experiment folder, so save pkl file there too
    local experiment_dir="$work_dir"
    local work_dir_basename=$(basename "$work_dir")
    # If work_dir is a timestamp subdirectory, use parent directory (experiment folder)
    if [[ "$work_dir_basename" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
        experiment_dir=$(dirname "$work_dir")
    fi
    local dump_path="${experiment_dir}/${DUMP_FILE}"
    
    # Check if dump file already exists
    if [ -f "$dump_path" ]; then
        write_log "----------------------------------------" "INFO"
        write_log "Dump file already exists: ${dump_path}" "INFO"
        write_log "Skipping test for [${index}/${total}]" "INFO"
        write_log "========================================" "INFO"
        return 0
    fi
    
    # Build test command
    local test_cmd="python -X utf8 tools/test.py \"${config_file}\" \"${checkpoint}\" --dump \"${dump_path}\""
    
    if [ ${#TEST_OPTIONS[@]} -gt 0 ]; then
        test_cmd="${test_cmd} ${TEST_OPTIONS[*]}"
    fi
    
    # Create separate log file for each test job
    local job_log_file="${LOG_DIR}/test_job_${index}_${TIMESTAMP}.log"
    
    write_log "Command: ${test_cmd}" "INFO"
    write_log "Dump file: ${dump_path}" "INFO"
    write_log "Job log: ${job_log_file}" "INFO"
    write_log "----------------------------------------" "INFO"
    
    # Execute test
    local start_time=$(date +%s)
    
    if [ "$is_background" = "true" ]; then
        # Background execution (parallel mode)
        # Output is saved to log file, can be viewed with: tail -f "$job_log_file"
        # Use bash -c to ensure the process is a direct child of this shell
        bash -c "$test_cmd" >> "$job_log_file" 2>&1 &
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
            write_log "Test output:" "INFO"
            write_log "----------------------------------------" "INFO"
            if eval "$test_cmd" 2>&1 | tee "$job_log_file"; then
                local exit_code=${PIPESTATUS[0]}
            else
                local exit_code=$?
            fi
        else
            # Silent execution (only log file)
            if eval "$test_cmd" >> "$job_log_file" 2>&1; then
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
            write_log "Testing completed: [${index}/${total}]" "INFO"
            write_log "Duration: ${duration_str}" "INFO"
            write_log "Dump file saved: ${dump_path}" "INFO"
            write_log "========================================" "INFO"
            
            return 0
        else
            write_log "----------------------------------------" "ERROR"
            write_log "Error occurred: [${index}/${total}]" "ERROR"
            write_log "Error message: Test command failed with exit code ${exit_code}" "ERROR"
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

write_log "Sequential/Parallel testing script started" "INFO"
write_log "Target epoch: ${TARGET_EPOCH}" "INFO"
write_log "Work directory: ${WORK_DIR_BASE}" "INFO"
write_log "Log file: ${LOG_FILE}" "INFO"
write_log ""

# Discover completed runs
write_log "=== Discovering Completed Training Runs ===" "INFO"
completed_runs_str=$(discover_completed_runs "$WORK_DIR_BASE" "$TARGET_EPOCH")
if [ -z "$completed_runs_str" ]; then
    write_log "No completed training runs found." "WARNING"
    exit 0
fi

# Parse completed runs
declare -a completed_runs_array
while IFS= read -r line; do
    if [ -n "$line" ]; then
        completed_runs_array+=("$line")
    fi
done <<< "$completed_runs_str"

num_jobs=${#completed_runs_array[@]}
write_log "Found ${num_jobs} completed training run(s)" "INFO"
write_log ""

# Decide execution mode
EXEC_MODE=$(decide_execution_mode $num_jobs)
write_log "Execution mode: ${EXEC_MODE}" "INFO"
write_log ""

success_count=0
fail_count=0
failed_runs=()
background_pids=()

write_log "=== Starting Testing ===" "INFO"
write_log ""

# Process each completed run
index=0
for run_info in "${completed_runs_array[@]}"; do
    ((index++))
    
    # Parse work_dir and checkpoint
    work_dir=$(echo "$run_info" | cut -d'|' -f1)
    checkpoint=$(echo "$run_info" | cut -d'|' -f2)
    
    # Find config file
    config_file=$(find_config_file "$work_dir")
    if [ -z "$config_file" ] || [ ! -f "$config_file" ]; then
        write_log "Warning: Could not find config file for ${work_dir}, skipping..." "WARNING"
        ((fail_count++))
        failed_runs+=("${work_dir} (config not found)")
        continue
    fi
    
    if [ "$EXEC_MODE" = "parallel" ]; then
        # Parallel execution
        pid=$(start_testing "$work_dir" "$checkpoint" "$config_file" "$index" "$num_jobs" "true")
        if [ $? -eq 0 ]; then
            background_pids+=("$pid")
            write_log "Started test job [${index}/${num_jobs}] with PID ${pid}" "INFO"
        else
            ((fail_count++))
            failed_runs+=("${work_dir}")
        fi
        sleep 2
    else
        # Sequential execution
        if start_testing "$work_dir" "$checkpoint" "$config_file" "$index" "$num_jobs" "false"; then
            ((success_count++))
        else
            ((fail_count++))
            failed_runs+=("${work_dir}")
        fi
        
        if [ $index -lt $num_jobs ]; then
            write_log "Waiting 5 seconds before next test..." "INFO"
            sleep 5
        fi
    fi
done

# Wait for parallel jobs
if [ "$EXEC_MODE" = "parallel" ] && [ ${#background_pids[@]} -gt 0 ]; then
    if wait_for_jobs "${background_pids[@]}"; then
        success_count=${#background_pids[@]}
    else
        fail_count=$((fail_count + ${#failed_runs[@]}))
    fi
fi

# ============================================
# Final Summary
# ============================================

write_log ""
write_log "========================================" "INFO"
write_log "Testing completed" "INFO"
write_log "========================================" "INFO"
write_log "Execution mode: ${EXEC_MODE}" "INFO"
write_log "Success: ${success_count} / ${num_jobs}" "INFO"
write_log "Failed: ${fail_count} / ${num_jobs}" "INFO"

if [ ${#failed_runs[@]} -gt 0 ]; then
    write_log ""
    write_log "Failed runs:" "WARNING"
    for failed_run in "${failed_runs[@]}"; do
        write_log "  - ${failed_run}" "WARNING"
    done
fi

write_log ""
write_log "Main log file: ${LOG_FILE}" "INFO"
if [ "$EXEC_MODE" = "parallel" ]; then
    write_log "Individual job logs: ${LOG_DIR}/test_job_*_${TIMESTAMP}.log" "INFO"
fi
write_log "========================================" "INFO"

# Return exit code 1 if any job failed
if [ $fail_count -gt 0 ]; then
    exit 1
else
    exit 0
fi

