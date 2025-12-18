# mmpose Sequential Training Script (PowerShell)
# Template for training multiple config files sequentially

# UTF-8 encoding settings
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# ============================================
# Configuration Area - Modify here
# ============================================

# List of config files to train (executed in order)
$CONFIG_FILES = @(
    "configs/body_2d_keypoint/custom/td-hm-hrnet-w48-adam-lr1e-2-warm100batch-bs8-ep100-coco-384x288_AP-base256.py",
    "configs/body_2d_keypoint/custom/td-hm_ViTPose-base_1xb8-100e_foot-ap-384x288.py",
    "configs/body_2d_keypoint/custom/td-hm_ViTPose-base_1xb16-100e_foot-ap-256x192.py"
    # Add more as needed
)

# Additional training options (modify if needed)
$TRAIN_OPTIONS = @(
    # "--amp",  # Enable automatic mixed precision training
    # "--resume",  # Auto resume from latest checkpoint
    # "--no-validate"  # Disable validation (not recommended)
)

# Log file directory
$LOG_DIR = "logs"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_FILE = Join-Path $LOG_DIR "train_sequential_$TIMESTAMP.log"

# ============================================
# Script Start
# ============================================

# Set current directory to mmpose directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $SCRIPT_DIR

# Create log directory
if (!(Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null
}

# Logging function
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] [$Level] $Message"
    Write-Host $LogMessage
    Add-Content -Path $LOG_FILE -Value $LogMessage
}

# Training execution function
function Start-Training {
    param(
        [string]$ConfigFile,
        [int]$Index,
        [int]$Total
    )
    
    Write-Log "========================================" "INFO"
    Write-Log "Training started: [$Index/$Total] $ConfigFile" "INFO"
    Write-Log "========================================" "INFO"
    
    # Check if config file exists
    if (!(Test-Path $ConfigFile)) {
        Write-Log "Error: Config file not found: $ConfigFile" "ERROR"
        return $false
    }
    
    # Build training command
    $TrainCmd = "python -X utf8 tools/train.py `"$ConfigFile`""
    
    if ($TRAIN_OPTIONS.Count -gt 0) {
        $TrainCmd += " " + ($TRAIN_OPTIONS -join " ")
    }
    
    Write-Log "Command: $TrainCmd" "INFO"
    Write-Log "----------------------------------------" "INFO"
    
    try {
        # Execute training
        $StartTime = Get-Date
        Invoke-Expression $TrainCmd
        
        $EndTime = Get-Date
        $Duration = $EndTime - $StartTime
        $DurationStr = "{0:D2}:{1:D2}:{2:D2}" -f $Duration.Hours, $Duration.Minutes, $Duration.Seconds
        
        Write-Log "----------------------------------------" "INFO"
        Write-Log "Training completed: [$Index/$Total] $ConfigFile" "INFO"
        Write-Log "Duration: $DurationStr" "INFO"
        Write-Log "========================================" "INFO"
        
        return $true
    }
    catch {
        Write-Log "Error occurred: [$Index/$Total] $ConfigFile" "ERROR"
        Write-Log "Error message: $($_.Exception.Message)" "ERROR"
        Write-Log "========================================" "ERROR"
        return $false
    }
}

# ============================================
# Main Execution
# ============================================

Write-Log "Sequential training script started" "INFO"
Write-Log "Total training jobs: $($CONFIG_FILES.Count)" "INFO"
Write-Log "Log file: $LOG_FILE" "INFO"
Write-Log ""

$SuccessCount = 0
$FailCount = 0
$FailedConfigs = @()

for ($i = 0; $i -lt $CONFIG_FILES.Count; $i++) {
    $ConfigFile = $CONFIG_FILES[$i]
    $Index = $i + 1
    $Total = $CONFIG_FILES.Count
    
    $Success = Start-Training -ConfigFile $ConfigFile -Index $Index -Total $Total
    
    if ($Success) {
        $SuccessCount++
    }
    else {
        $FailCount++
        $FailedConfigs += $ConfigFile
    }
    
    # Wait before next training if not the last job
    if ($i -lt $CONFIG_FILES.Count - 1) {
        Write-Log "Waiting 10 seconds before next training..." "INFO"
        Start-Sleep -Seconds 10
    }
}

# ============================================
# Final Summary
# ============================================

Write-Log ""
Write-Log "========================================" "INFO"
Write-Log "Sequential training completed" "INFO"
Write-Log "========================================" "INFO"
Write-Log "Success: $SuccessCount / $($CONFIG_FILES.Count)" "INFO"
Write-Log "Failed: $FailCount / $($CONFIG_FILES.Count)" "INFO"

if ($FailedConfigs.Count -gt 0) {
    Write-Log ""
    Write-Log "Failed config files:" "WARNING"
    foreach ($FailedConfig in $FailedConfigs) {
        Write-Log "  - $FailedConfig" "WARNING"
    }
}

Write-Log ""
Write-Log "Log file: $LOG_FILE" "INFO"
Write-Log "========================================" "INFO"

# Return exit code 1 if any job failed
if ($FailCount -gt 0) {
    exit 1
}
else {
    exit 0
}

