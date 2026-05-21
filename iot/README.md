# Forest Audio AI: IoT Architecture

## Overview
The Forest Audio AI IoT system is a hierarchical network designed for resilient forest monitoring in remote areas of Moldova. It consists of multiple low-power sensor nodes that capture audio and environmental data, communicating with a central gateway that manages cloud connectivity and pre-processing.

## Network Topology
```text
+-------------------+       +-------------------+       +-------------------+
|  ESP32 Node 01    |       |  ESP32 Node 02    |       |  ESP32 Node 03    |
| (Mic + Env Sens)  |       | (Mic + Env Sens)  |       | (Mic + Env Sens)  |
+---------+---------+       +---------+---------+       +---------+---------+
          |                           |                           |
          |           WiFi / LoRa     |                           |
          +---------------------------+---------------------------+
                                      |
                                      v
                        +---------------------------+
                        |   Raspberry Pi Gateway    |
                        | (MQTT Broker + Edge Logic)|
                        +-------------+-------------+
                                      |
                                      | HTTPS / WebSocket
                                      v
                        +---------------------------+
                        |      Cloud API Server     |
                        |  (Inference + Dashboard)  |
                        +---------------------------+
```

## Component Roles
1. **ESP32 Sensor Nodes**:
   - **Audio Capture**: Uses I2S microphones to record 5-second high-fidelity audio chunks.
   - **Environment Sensing**: Captures temperature, humidity, and pressure.
   - **Edge Filtering**: Performs basic energy thresholding to skip transmission of silent chunks.
   - **Local Storage**: Buffers data to an SD card if communication with the gateway is lost.

2. **Raspberry Pi Gateway**:
   - **Message Broker**: Runs an MQTT broker (e.g., Mosquitto) to receive data from nodes.
   - **Data Aggregation**: Queues incoming audio and metadata for transmission to the cloud.
   - **Connectivity Management**: Monitors internet availability and handles retries for failed cloud requests.
   - **Health Monitoring**: Tracks sensor node heartbeats and packet loss rates.

3. **Cloud API Server**:
   - **Inference Pipeline**: Runs the CRNN and XGBoost models to detect threats and assess risk.
   - **Alerting**: Triggers WebSocket events and notifications for high-risk detections.

## Data Flow
1. **Capture**: ESP32 records 5s of audio @ 32kHz.
2. **Threshold**: Node calculates RMS energy. If below threshold, chunk is discarded to save power.
3. **Transmit**: Node publishes audio (Base64) + metadata via MQTT to the Gateway.
4. **Buffer**: Gateway receives MQTT message and adds it to a processing queue.
5. **Forward**: Gateway sends data to `/api/pipeline/analyze` via HTTPS.
6. **Inference**: Cloud API processes data and broadcasts results via WebSocket.

## Connectivity Handling Strategy
- **Node-to-Gateway Loss**: Nodes store data in an "Offline Queue" on an internal SD card. Once the connection is restored, they resume streaming live data while slowly draining the offline queue in the background.
- **Gateway-to-Cloud Loss**: The Gateway buffers failed API requests to local storage. It uses an exponential backoff strategy to retry requests once internet connectivity is restored. High-priority alerts (based on local pre-analysis) are queued first.
