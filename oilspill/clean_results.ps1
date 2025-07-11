# PowerShell script to clean up all data generated by main_train.py

Write-Host "Cleaning up training data generated by main_train.py..." -ForegroundColor Yellow

# Define paths to clean
$pathsToClean = @(
    "runs",
    "logs", 
    "saved_models",
    "episode_gifs",
    "__pycache__",
    "agents/__pycache__",
    "data_generation/__pycache__",
    "environments/__pycache__"
)

# Function to safely remove directory
function Remove-DirectorySafely {
    param($path)
    
    if (Test-Path $path) {
        try {
            Write-Host "Removing: $path" -ForegroundColor Red
            Remove-Item -Path $path -Recurse -Force
            Write-Host "Successfully removed: $path" -ForegroundColor Green
        }
        catch {
            Write-Host "Failed to remove $path : $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    else {
        Write-Host "Path does not exist: $path" -ForegroundColor Gray
    }
}

# Clean each directory
foreach ($path in $pathsToClean) {
    Remove-DirectorySafely -path $path
}

# Also clean any temp_saved_models directory if it exists
if (Test-Path "temp_saved_models") {
    Remove-DirectorySafely -path "temp_saved_models"
}

# Clean any .pyc files that might be scattered
Write-Host "Removing any remaining .pyc files..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Filter "*.pyc" | Remove-Item -Force

Write-Host "Cleanup complete!" -ForegroundColor Green
Write-Host "The following directories have been cleaned:" -ForegroundColor Cyan
$pathsToClean | ForEach-Object { Write-Host "  - $_" -ForegroundColor White }