# Auxiliary function to stop on failure
function Invoke-Call {
    param (
        [scriptblock]$ScriptBlock,
        [string]$ErrorAction = $ErrorActionPreference
    )
    & @ScriptBlock
    if (($lastexitcode -ne 0) -and $ErrorAction -eq "Stop") {
        Pop-Location
        exit $lastexitcode
    }
}

# Activate env
Invoke-Call -ScriptBlock { python -m venv . } -ErrorAction Stop
Invoke-Call -ScriptBlock { ./Scripts/Activate.ps1 } -ErrorAction Stop

# Install requirements
Invoke-Call -ScriptBlock { poetry install } -ErrorAction Stop

# Linter checks
Invoke-Call -ScriptBlock { pre-commit install } -ErrorAction Stop
Invoke-Call -ScriptBlock { pre-commit run -a } -ErrorAction Stop

# Create new directories
Invoke-Call -ScriptBlock { New-Item -ItemType Directory dataset } -ErrorAction Ignore
Invoke-Call -ScriptBlock { New-Item -ItemType Directory saved_model } -ErrorAction Ignore

# Train and infer model
Invoke-Call -ScriptBlock { python train.py } -ErrorAction Stop
Invoke-Call -ScriptBlock { python infer.py } -ErrorAction Stop
