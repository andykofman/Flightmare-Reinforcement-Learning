# ArduPilot Setup on Windows - Quick Start Guide

## Your Current Situation
- Board connected as: **COM3** on Windows
- Docker container can't see COM ports (normal behavior)

## Solution: Test from Windows First

### Step 1: Install MAVProxy on Windows

**Option A: Using Python (if you have Python installed)**
```powershell
pip install pymavlink mavproxy
```

**Option B: Using QGroundControl (Easier - GUI tool)**
1. Download from: https://qgroundcontrol.com/downloads
2. Install and run
3. Connect to COM3 directly from the GUI

### Step 2: Test Connection with MAVProxy

Open PowerShell or Command Prompt:

```powershell
# Test connection
mavproxy.py --master=COM3 --baudrate=115200

# If that doesn't work, try different baud rates:
mavproxy.py --master=COM3 --baudrate=57600
mavproxy.py --master=COM3 --baudrate=921600
```

**What you should see:**
```
Connect COM3:115200
Waiting for heartbeat from COM3:115200
Heartbeat from system 1
```

### Step 3: Test with QGroundControl (Easiest)

1. Open QGroundControl
2. Click "Q" icon → Vehicle Setup
3. Select "Serial" connection
4. Choose COM3
5. Set baud rate: 115200 (or try 57600, 921600)
6. Click "Connect"

**If it connects:**
- You'll see vehicle status
- Can configure parameters
- Can see sensor data

### Step 4: Configure ArduPilot for HIL

In QGroundControl or Mission Planner:

1. Go to **Parameters** (wrench icon)
2. Search for these parameters:
   - `HIL_MODE` → Set to `1` (HIL Simulation)
   - `SIM_SPEEDUP` → Set to `1`
   - `HIL_SERVOS` → Set to `1`
3. Click "Write Parameters"
4. Reboot board (disconnect/reconnect USB)

## Next: Access from Docker Container

Once connection works from Windows, we'll set up Docker access.

