# Forest Audio AI: IoT Deployment Guide

This guide provides instructions for setting up and deploying the sensor nodes and gateway for the Forest Audio AI system.

## Hardware Requirements

### Sensor Node (per node)
- **Microcontroller**: ESP32-S3 (with 8MB PSRAM recommended)
- **Microphone**: INMP441 I2S Digital Microphone
- **Environmental Sensor**: BME280 (Temperature, Humidity, Pressure)
- **Storage**: MicroSD Card Module + 16GB Class 10 SD Card
- **Power**: 3.7V LiPo Battery (2000mAh+) or Solar Panel with Charge Controller

### Gateway
- **Single Board Computer**: Raspberry Pi 4B (4GB RAM minimum)
- **Storage**: 32GB MicroSD Card
- **Power**: 5V 3A USB-C Power Supply
- **Network**: WiFi or Ethernet for Internet connectivity

## Network Requirements
- **Local Network**: WiFi (2.4GHz) or LoRa for Node-to-Gateway communication.
- **Internet**: Broadband or Cellular (4G/LTE) for Gateway-to-Cloud communication.
- **Protocols**: MQTT (port 1883), HTTPS (port 443).

## Setup Instructions

### 1. Gateway Setup
1. Install Raspberry Pi OS (Lite recommended).
2. Install Mosquitto MQTT Broker:
   ```bash
   sudo apt update && sudo apt install mosquitto mosquitto-clients
   sudo systemctl enable mosquitto
   ```
3. Deploy the `gateway.py` script and install dependencies:
   ```bash
   pip install requests paho-mqtt
   ```
4. Update `GATEWAY_CONFIG` in `gateway.py` with your Cloud API URL.
5. Run the gateway:
   ```bash
   python gateway.py
   ```

### 2. Sensor Node Setup
1. Wire the INMP441 microphone to the ESP32 (I2S pins: SCK, WS, SD).
2. Wire the BME280 sensor to the ESP32 (I2C pins: SDA, SCL).
3. Flash the firmware (based on `sensor_node.pseudo.py`) using ESP-IDF or Arduino IDE.
4. Configure the `SENSOR_ID` and `GATEWAY_IP` in the node settings.
5. Power on the node and verify connectivity via the Gateway logs.

## Troubleshooting

### Common Failure Modes
- **High Packet Loss**: Check WiFi signal strength or switch to LoRa for denser forest areas.
- **Silent Transmissions**: Adjust the `ENERGY_THRESHOLD` in the node firmware if valid sounds are being skipped.
- **Gateway Offline**: Ensure the Pi is properly powered and the MQTT service is running (`systemctl status mosquitto`).
- **Cloud API Errors**: Verify the `CLOUD_API_URL` and check for network firewalls blocking outbound HTTPS traffic.

### Fixes
- **SD Card Full**: Nodes automatically rotate log files, but ensure class 10 cards are used to prevent write latency issues.
- **Battery Drain**: Increase the sleep interval between audio captures if using solar power in low-light seasons.
