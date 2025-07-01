# PowerShell script to copy all .py, .yaml, and .json files from the oilspill folder to py_files folder

# Define source and destination directories
$sourceDir = "c:\Users\Pedro\Documents\MAI\TFM\MAI-TFM\oilspill"
$destinationDir = "c:\Users\Pedro\Documents\MAI\TFM\MAI-TFM\py_files"

# Create destination directory if it doesn't exist
if (!(Test-Path -Path $destinationDir)) {
    New-Item -ItemType Directory -Path $destinationDir
}

# Copy all .py, .yaml, and .json files recursively from source to destination
Get-ChildItem -Path $sourceDir -Recurse -Include "*.py", "*.yaml", "*.json" | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $destinationDir
}

Write-Host "All .py, .yaml, and .json files have been copied to $destinationDir."
