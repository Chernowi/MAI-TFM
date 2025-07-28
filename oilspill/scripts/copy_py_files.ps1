# This script copies all necessary Python source files from the 'marl_framework'
# and the root directory into the 'py_files' directory for easier packaging or analysis.

# Get the project root directory, assuming this script is located in the 'scripts' folder.
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

# Define source paths and the destination directory
$sourceFrameworkDir = Join-Path $projectRoot "marl_framework"
$sourceMainTrainFile = Join-Path $projectRoot "main_train.py"
$destinationDir = Join-Path $projectRoot "py_files"

Write-Host "Starting file copy process..." -ForegroundColor Yellow

# 1. Create the destination directory if it doesn't exist
if (-not (Test-Path -Path $destinationDir)) {
    Write-Host "Creating destination directory: $destinationDir"
    New-Item -ItemType Directory -Path $destinationDir | Out-Null
} else {
    Write-Host "Destination directory '$destinationDir' already exists. Files will be overwritten."
}

# 2. Copy all .py files from the marl_framework directory
if (Test-Path $sourceFrameworkDir) {
    Write-Host "Copying all .py files from '$sourceFrameworkDir'..."
    Get-ChildItem -Path $sourceFrameworkDir -Recurse -Filter "*.py" | Copy-Item -Destination $destinationDir -Force
    Write-Host "Successfully copied files from marl_framework." -ForegroundColor Green
} else {
    Write-Host "Warning: Source directory '$sourceFrameworkDir' not found." -ForegroundColor Red
}

# 3. Copy main_train.py from the project root
if (Test-Path $sourceMainTrainFile) {
    Write-Host "Copying '$sourceMainTrainFile'..."
    Copy-Item -Path $sourceMainTrainFile -Destination $destinationDir -Force
    Write-Host "Successfully copied main_train.py." -ForegroundColor Green
} else {
    Write-Host "Warning: Source file '$sourceMainTrainFile' not found." -ForegroundColor Red
}

Write-Host "Script finished. All specified Python files have been copied to '$destinationDir'." -ForegroundColor Cyan