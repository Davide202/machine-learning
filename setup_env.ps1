# setup_env.ps1
# Script to configure the environment for the machine-learning project on a new Windows PC

Write-Host "--- Configuring Windows PC for Python with uv ---" -ForegroundColor Cyan

# 1. Update Execution Policy to allow running local scripts like the .venv activation
$currentPolicy = Get-ExecutionPolicy
if ($currentPolicy -eq "Restricted") {
    Write-Host "Setting ExecutionPolicy to RemoteSigned (CurrentUser scope)..." -ForegroundColor Yellow
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
} else {
    Write-Host "ExecutionPolicy is already set to $currentPolicy (sufficient for running scripts)." -ForegroundColor Green
}

# 2. Install uv if not currently installed
if (!(Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "Installing 'uv' (Python package manager) from astral.sh..." -ForegroundColor Yellow
    Invoke-RestMethod -Uri "https://astral.sh/uv/install.ps1" | Invoke-Expression
    
    # Reload environment variables for the current session to make uv available immediately
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
} else {
    Write-Host "'uv' is already installed." -ForegroundColor Green
}

# 3. Ensure Python is installed via uv (will fetch standard CPython if missing)
Write-Host "Setting up Python via uv..." -ForegroundColor Yellow
uv python install

# 4. Create virtual environment if it does not exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv..." -ForegroundColor Yellow
    uv venv
} else {
    Write-Host "Virtual environment (.venv) already exists." -ForegroundColor Green
}

# 5. Provide activation and run instructions
Write-Host "----------------------------------------------------" -ForegroundColor Cyan
Write-Host "Configuration completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment, run the following command in PowerShell:" -ForegroundColor White
Write-Host ".venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "----------------------------------------------------" -ForegroundColor Cyan
