# Accessing COM Ports from Docker on Windows

## The Problem
Docker containers on Windows can't directly access COM ports (like COM3).

## Solutions

### Solution 1: Use WSL2 with USB Passthrough (Recommended if using WSL2)

If your Docker Desktop uses WSL2 backend:

1. **Install USBIP in WSL2:**
```bash
# In WSL2 terminal
sudo apt-get update
sudo apt-get install usbip hwdata usbutils
sudo modprobe usbip-core
sudo modprobe usbip-host
```

2. **Find your device:**
```powershell
# In Windows PowerShell (as Administrator)
usbipd list
# You should see your ArduPilot device
```

3. **Share the device:**
```powershell
# In Windows PowerShell (as Administrator)
usbipd bind --busid <BUSID>
# Replace <BUSID> with the ID from step 2
```

4. **Attach in WSL2:**
```bash
# In WSL2 terminal
sudo usbip attach -r localhost -b <BUSID>
```

5. **Now it should appear in WSL2:**
```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

6. **Run Docker with device access:**
```bash
# Stop current container
docker stop flightmare

# Run with device access
docker run -it --rm \
  --device=/dev/ttyUSB0 \
  -v $(pwd):/root/flightmare \
  -p 10253:10253 -p 10254:10254 \
  flightmare:latest
```

### Solution 2: Use Windows COM Port Bridge (Easier)

Use a tool to bridge COM port to network:

1. **Install com0com (Virtual Serial Port):**
   - Download: https://sourceforge.net/projects/com0com/
   - Creates virtual COM ports

2. **Or use TCP/IP Bridge:**
   - Use `socat` or similar to bridge COM3 to TCP
   - Connect Docker container to TCP port

### Solution 3: Run ROS Directly on Windows (Simplest for Testing)

Skip Docker for ArduPilot testing:

1. **Install ROS on Windows:**
   - Use ROS2 (has Windows support)
   - Or use WSL2 with ROS directly (not Docker)

2. **Test connection:**
```bash
# In WSL2 (not Docker)
mavproxy.py --master=/dev/ttyS3 --baudrate=115200
# COM3 in Windows = /dev/ttyS3 in WSL2
```

### Solution 4: Use Network MAVLink (Best for HIL)

Instead of USB, use UDP/TCP:

1. **On Windows, bridge COM3 to UDP:**
```powershell
# Use MAVProxy to bridge
mavproxy.py --master=COM3 --out=udp:127.0.0.1:14550
```

2. **In Docker, connect to UDP:**
```bash
# Docker can access network ports easily
mavproxy.py --master=udp:127.0.0.1:14550
```

## Recommended Approach

**For initial testing:** Use Solution 3 (WSL2 directly) or Solution 4 (Network bridge)

**For production:** Use Solution 1 (USB passthrough) or Solution 4 (Network)




