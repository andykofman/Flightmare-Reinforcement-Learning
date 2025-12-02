# Quick Start: Connect COM3 to Docker

## Simplest Solution: Network Bridge

Since Docker can't directly access COM3, we'll bridge it to a network port.

### Step 1: Bridge COM3 to Network (Run on Windows)

**Open PowerShell (as Administrator):**

```powershell
# Install MAVProxy if not already installed
pip install pymavlink mavproxy

# Bridge COM3 to UDP port (run this and keep it running)
mavproxy.py --master=COM3 --baudrate=115200 --out=udp:127.0.0.1:14550
```

**What this does:**
- Reads from COM3 (your ArduPilot board)
- Sends to UDP port 14550 (Docker can access this)

**Keep this window open!** It needs to keep running.

### Step 2: Test Connection from Docker

**In your Docker container:**

```bash
# Install MAVProxy in container (if not already)
pip install pymavlink mavproxy

# Connect to the UDP bridge
mavproxy.py --master=udp:127.0.0.1:14550
```

**You should see:**
```
Connect udp:127.0.0.1:14550
Waiting for heartbeat...
Heartbeat from system 1
```

### Step 3: Test with ROS/MAVROS

**Update docker-compose.yml to expose UDP port:**

Add this to `docker-compose.yml`:
```yaml
ports:
  - "10253:10253"
  - "10254:10254"
  - "14550:14550/udp"  # Add this line
```

**Then in Docker, create test launch file:**

Create: `flightmare/flightros/launch/test_ardupilot_udp.launch`
```xml
<launch>
  <node pkg="mavros" type="mavros_node" name="mavros" output="screen">
    <param name="fcu_url" value="udp://127.0.0.1:14550@14540"/>
    <param name="gcs_url" value="udp://@127.0.0.1:14550"/>
  </node>
</launch>
```

**Run it:**
```bash
# In Docker container
roscore &
roslaunch flightros test_ardupilot_udp.launch
```

## Alternative: Use WSL2 Directly (No Docker)

If Docker is too complex, test directly in WSL2:

1. **In WSL2, COM3 appears as `/dev/ttyS3`:**
```bash
# In WSL2 terminal (not Docker)
mavproxy.py --master=/dev/ttyS3 --baudrate=115200
```

2. **Or use Windows path:**
```bash
# WSL2 can access Windows COM ports
mavproxy.py --master=COM3 --baudrate=115200
```

## Troubleshooting

**"Permission denied" on /dev/ttyS3:**
```bash
sudo chmod 666 /dev/ttyS3
```

**"Device not found":**
- Make sure board is connected
- Try different baud rates: 57600, 115200, 921600
- Check Windows Device Manager for correct COM port

**"Connection timeout":**
- Make sure bridge is running (Step 1)
- Check firewall isn't blocking UDP port 14550
- Try different port: `--out=udp:127.0.0.1:14551`

