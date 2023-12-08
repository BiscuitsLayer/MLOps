# Auxiliary function to stop on failure
function Invoke-Call {
    param (
        [scriptblock]$ScriptBlock,
        [string]$ErrorAction = $ErrorActionPreference
    )
    & @ScriptBlock
    if (($lastexitcode -ne 0) -and $ErrorAction -eq "Stop") {
        Pop-Location
        Write-host $ErrorAction
        exit $lastexitcode
    }
    if ($lastexitcode -ne 0) {
        Write-host $ErrorAction
    }
}

# Activate env
Invoke-Call -ScriptBlock { python -m venv . } -ErrorAction Ignore
Invoke-Call -ScriptBlock { ./Scripts/Activate.ps1 } -ErrorAction Ignore

# Install requirements
Invoke-Call -ScriptBlock { poetry install } -ErrorAction Stop

# Linter checks
Invoke-Call -ScriptBlock { pre-commit install } -ErrorAction Stop
Invoke-Call -ScriptBlock { pre-commit run -a } -ErrorAction Stop

# Create new directories
New-Item -ItemType Directory dataset -ErrorAction Ignore
New-Item -ItemType Directory saved_model -ErrorAction Ignore

# Train and infer model
Invoke-Call -ScriptBlock { poetry run mnist_train run_train } -ErrorAction Stop
Invoke-Call -ScriptBlock { poetry run mnist_infer run_infer } -ErrorAction Stop
