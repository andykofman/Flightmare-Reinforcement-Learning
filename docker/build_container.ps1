#!/usr/bin/env pwsh
param(
    [switch]$Gpu,
    [switch]$NoRos,
    [switch]$NoRl,
    [switch]$Push,
    [string]$Tag = "latest",
    [string]$PytorchVersion = "2.0.0",
    [string]$CudaVersion = "cpu",
    [switch]$Help
)

if ($Help) {
    Write-Host "Flightmare Docker Build Script"
    Write-Host "Usage: .\build_container.ps1 [-Gpu] [-NoRos] [-NoRl] [-Push] [-Tag <tag>]"
    exit 0
}

$IMAGE_NAME = "flightmare"
$IMAGE_TAG = $Tag
$FULL_IMAGE_NAME = "${IMAGE_NAME}:${IMAGE_TAG}"

if ($Gpu -and $CudaVersion -eq "cpu") {
    $CudaVersion = "cu118"
}

$ENABLE_ROS = if ($NoRos) { "0" } else { "1" }
$ENABLE_RL = if ($NoRl) { "0" } else { "1" }

Write-Host ""
Write-Host "Building $FULL_IMAGE_NAME" -ForegroundColor Cyan
Write-Host "  GPU: $(if ($Gpu) { 'YES' } else { 'NO' })"
Write-Host "  ROS: $(if ($NoRos) { 'NO' } else { 'YES' })"
Write-Host "  RL:  $(if ($NoRl) { 'NO' } else { 'YES' })"
Write-Host ""

Write-Host "[1/3] Checking Docker..." -ForegroundColor Yellow
docker --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker not found" -ForegroundColor Red
    exit 1
}

Write-Host "[2/3] Building image..." -ForegroundColor Yellow
docker build `
    --build-arg PYTORCH_VERSION=$PytorchVersion `
    --build-arg CUDA_VERSION=$CudaVersion `
    --build-arg ENABLE_ROS=$ENABLE_ROS `
    --build-arg ENABLE_RL=$ENABLE_RL `
    -t $FULL_IMAGE_NAME `
    -f Dockerfile `
    ..

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed" -ForegroundColor Red
    exit 1
}

Write-Host "[3/3] Verifying..." -ForegroundColor Yellow
docker run --rm $FULL_IMAGE_NAME /bin/bash /root/verify_installation.sh

if ($Push) {
    Write-Host "Pushing to registry..." -ForegroundColor Yellow
    docker push $FULL_IMAGE_NAME
}

Write-Host ""
Write-Host "SUCCESS! Image: $FULL_IMAGE_NAME" -ForegroundColor Green
Write-Host ""
Write-Host "Run: docker run -it --rm $FULL_IMAGE_NAME" -ForegroundColor Cyan
Write-Host ""
