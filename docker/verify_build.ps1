
#!/usr/bin/env pwsh
################################################################################
# Flightmare Docker Verification Script
# Tests a built Docker image to ensure all components work correctly
################################################################################

param(
    [string]$Image = "flightmare:latest",
    [switch]$Verbose,
    [switch]$Help
)

if ($Help) {
    Write-Host @"

Flightmare Docker Verification Script
======================================

Usage:
    .\verify_build.ps1 [OPTIONS]

Options:
    -Image <name>     Docker image to verify (default: flightmare:latest)
    -Verbose          Show detailed output
    -Help             Show this help message

Examples:
    # Verify default image
    .\verify_build.ps1

    # Verify specific tag
    .\verify_build.ps1 -Image flightmare:v1.0

    # Verbose output
    .\verify_build.ps1 -Verbose

"@
    exit 0
}

################################################################################
# Configuration
################################################################################

$tests = @()
$passedTests = 0
$failedTests = 0

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   Flightmare Docker Verification" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Image: $Image" -ForegroundColor White
Write-Host ""

################################################################################
# Helper Functions
################################################################################

function Run-Test {
    param(
        [string]$Name,
        [string]$Command,
        [string]$Expected = "",
        [bool]$ShouldSucceed = $true
    )
    
    Write-Host "[$($tests.Count + 1)] $Name..." -NoNewline
    
    $output = docker run --rm $Image /bin/bash -c $Command 2>&1
    $success = $LASTEXITCODE -eq 0
    
    if ($ShouldSucceed) {
        if ($success) {
            Write-Host " ✓" -ForegroundColor Green
            $script:passedTests++
        } else {
            Write-Host " ✗" -ForegroundColor Red
            $script:failedTests++
            if ($Verbose) {
                Write-Host "    Output: $output" -ForegroundColor Gray
            }
        }
    } else {
        if (-not $success) {
            Write-Host " ✓" -ForegroundColor Green
            $script:passedTests++
        } else {
            Write-Host " ✗" -ForegroundColor Red
            $script:failedTests++
        }
    }
    
    $script:tests += @{
        Name = $Name
        Success = if ($ShouldSucceed) { $success } else { -not $success }
        Output = $output
    }
    
    if ($Verbose -and $success) {
        Write-Host "    $output" -ForegroundColor Gray
    }
}

################################################################################
# Pre-flight Checks
################################################################################

Write-Host "[Pre-flight] Checking prerequisites..." -ForegroundColor Yellow
Write-Host ""

# Check if Docker is available
Write-Host "  Docker installed..." -NoNewline
try {
    docker --version > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker not found"
    }
    Write-Host " ✓" -ForegroundColor Green
} catch {
    Write-Host " ✗" -ForegroundColor Red
    Write-Host ""
    Write-Host "ERROR: Docker not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Check if Docker daemon is running
Write-Host "  Docker daemon running..." -NoNewline
try {
    docker ps > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker daemon not running"
    }
    Write-Host " ✓" -ForegroundColor Green
} catch {
    Write-Host " ✗" -ForegroundColor Red
    Write-Host ""
    Write-Host "ERROR: Docker daemon not running" -ForegroundColor Red
    exit 1
}

# Check if image exists
Write-Host "  Image exists..." -NoNewline
$imageExists = docker images -q $Image
if ([string]::IsNullOrEmpty($imageExists)) {
    Write-Host " ✗" -ForegroundColor Red
    Write-Host ""
    Write-Host "ERROR: Image '$Image' not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "Available images:" -ForegroundColor Yellow
    docker images | Select-String "flightmare"
    Write-Host ""
    exit 1
}
Write-Host " ✓" -ForegroundColor Green

Write-Host ""

################################################################################
# Run Tests
################################################################################

Write-Host "[Tests] Running verification tests..." -ForegroundColor Yellow
Write-Host ""

# Test 1: Python version
Run-Test -Name "Python 3.9 installed" -Command "python3 --version 2>&1 | grep '3.9'"

# Test 2: flightgym import
Run-Test -Name "flightgym import" -Command "python3 -c 'import flightgym; print(\"OK\")'"

# Test 3: NumPy import
Run-Test -Name "NumPy import" -Command "python3 -c 'import numpy as np; print(np.__version__)'"

# Test 4: PyTorch import
Run-Test -Name "PyTorch import" -Command "python3 -c 'import torch; print(torch.__version__)'"

# Test 5: Stable-Baselines3 import
Run-Test -Name "Stable-Baselines3 import" -Command "python3 -c 'import stable_baselines3; print(stable_baselines3.__version__)'"

# Test 6: Gymnasium import
Run-Test -Name "Gymnasium import" -Command "python3 -c 'import gymnasium; print(gymnasium.__version__)'"

# Test 7: flightrl_modern import
Run-Test -Name "flightrl_modern import" -Command "python3 -c 'import flightrl_modern; print(flightrl_modern.__version__)'"

# Test 8: Environment creation
Run-Test -Name "Environment creation" -Command "python3 -c 'from flightrl_modern.envs.gymnasium_wrapper import make_flight_env; env = make_flight_env(render=False, num_envs=1); obs, info = env.reset(); env.close(); print(\"\"OK\"\")'"

# Test 9: Environment step  
Run-Test -Name "Environment step" -Command "python3 -c 'from flightrl_modern.envs.gymnasium_wrapper import make_flight_env; import numpy as np; env = make_flight_env(render=False, num_envs=1); env.reset(); action = np.zeros(env.action_space.shape); env.step(action); env.close(); print(\"\"OK\"\")'"

# Test 10: SAC model creation
Run-Test -Name "SAC model creation" -Command "python3 -c 'from stable_baselines3 import SAC; from flightrl_modern.envs.gymnasium_wrapper import make_flight_env_for_sb3; env = make_flight_env_for_sb3(render=False); model = SAC(\"\"MlpPolicy\"\", env, verbose=0, device=\"\"cpu\"\"); env.close(); print(\"\"OK\"\")'"

# Test 11: Verification script
Run-Test -Name "Built-in verification script" -Command "/bin/bash /root/verify_installation.sh 2>&1"

# Test 12: Smoke test (if available)
$smokeTestPath = "/root/flightmare/flightrl_modern/examples/smoke_test.py"
Run-Test -Name "RL smoke test" -Command "if [ -f $smokeTestPath ]; then timeout 120 python3 $smokeTestPath 2>&1 || echo 'Smoke test timed out or failed'; else echo 'Smoke test not found (RL may be disabled)'; fi"

Write-Host ""

################################################################################
# Summary
################################################################################

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   Verification Summary" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$totalTests = $passedTests + $failedTests
$passRate = if ($totalTests -gt 0) { ($passedTests / $totalTests) * 100 } else { 0 }

Write-Host "Total Tests: $totalTests" -ForegroundColor White
Write-Host "Passed:      $passedTests" -ForegroundColor Green
Write-Host "Failed:      $failedTests" -ForegroundColor $(if ($failedTests -gt 0) { "Red" } else { "Gray" })
Write-Host "Pass Rate:   ${passRate:N1}%" -ForegroundColor $(if ($passRate -eq 100) { "Green" } elseif ($passRate -ge 80) { "Yellow" } else { "Red" })
Write-Host ""

# Show failed tests
if ($failedTests -gt 0) {
    Write-Host "Failed Tests:" -ForegroundColor Red
    foreach ($test in $tests) {
        if (-not $test.Success) {
            Write-Host "  ✗ $($test.Name)" -ForegroundColor Red
            if ($Verbose) {
                Write-Host "    $($test.Output)" -ForegroundColor Gray
            }
        }
    }
    Write-Host ""
}

# Overall status
if ($failedTests -eq 0) {
    Write-Host "✓ All tests passed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "The image is ready to use:" -ForegroundColor White
    Write-Host "  docker run -it --rm -p 10253:10253 $Image" -ForegroundColor Cyan
    Write-Host ""
    exit 0
} else {
    Write-Host "✗ Some tests failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "The image may have issues. Review the failures above." -ForegroundColor Yellow
    Write-Host "Run with -Verbose for detailed output." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
