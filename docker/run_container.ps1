#!/usr/bin/env pwsh
################################################################################
# Flightmare Docker Run Script
# Runs the Flightmare container with various options
################################################################################

param(
    [switch]$GPU,           # Enable GPU support (requires NVIDIA Docker)
    [switch]$Mount,         # Mount current directory to /workspace
    [switch]$Interactive,   # Run in interactive mode (default)
    [string]$Command = ""   # Command to run (if not interactive)
)

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Flightmare Docker Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$IMAGE_NAME = "flightmare:latest"

# Check if image exists
Write-Host "Checking for Docker image..." -NoNewline
$imageExists = docker images -q $IMAGE_NAME

if ([string]::IsNullOrEmpty($imageExists)) {
    Write-Host " ✗" -ForegroundColor Red
    Write-Host ""
    Write-Host "ERROR: Docker image '$IMAGE_NAME' not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please build the image first:" -ForegroundColor Yellow
    Write-Host "  .\build_container.ps1" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}
Write-Host " ✓" -ForegroundColor Green
Write-Host ""

# Build Docker run command
$dockerArgs = @("run")

# Interactive mode
if ($Interactive -or [string]::IsNullOrEmpty($Command)) {
    $dockerArgs += "-it"
}

# Always remove container after exit
$dockerArgs += "--rm"

# Port mapping for Unity Bridge
$dockerArgs += "-p", "10253:10253"
Write-Host "  Port mapping: 10253:10253 (Unity Bridge)" -ForegroundColor Gray

# GPU support
if ($GPU) {
    $dockerArgs += "--gpus", "all"
    Write-Host "  GPU support: Enabled" -ForegroundColor Gray
}

# Mount current directory
if ($Mount) {
    $currentDir = (Get-Location).Path
    $dockerArgs += "-v", "${currentDir}:/workspace"
    Write-Host "  Volume mount: $currentDir -> /workspace" -ForegroundColor Gray
}

# Add image name
$dockerArgs += $IMAGE_NAME

# Add command if specified
if (-not [string]::IsNullOrEmpty($Command)) {
    $dockerArgs += "bash", "-c", $Command
    Write-Host "  Command: $Command" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Starting container..." -ForegroundColor Yellow
Write-Host ""

# Run the container
& docker @dockerArgs

Write-Host ""
Write-Host "Container exited." -ForegroundColor Gray
Write-Host ""
