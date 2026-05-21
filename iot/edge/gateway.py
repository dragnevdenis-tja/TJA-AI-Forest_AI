import json
import time
import queue
import threading
import requests
import logging
from collections import deque
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ForestGateway")

# Configuration
MQTT_TOPIC = "forest/audio/+"
CLOUD_API_URL = "http://cloud-api-host/api/pipeline/analyze"
MAX_QUEUE_SIZE = 1000
RETRY_BUFFER_SIZE = 100

class Gateway:
    def __init__(self):
        self.data_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.retry_buffer = deque(maxlen=RETRY_BUFFER_SIZE)
        self.sensor_health = {} # sensor_id -> {"last_seen": ts, "packets": int, "errors": int}
        self.is_running = True

    def on_mqtt_message(self, sensor_id, payload):
        """Callback for incoming MQTT messages from sensor nodes."""
        try:
            data = json.loads(payload)
            self.data_queue.put(data, timeout=1)
            
            # Update health stats
            if sensor_id not in self.sensor_health:
                self.sensor_health[sensor_id] = {"last_seen": None, "packets": 0, "errors": 0}
            
            stats = self.sensor_health[sensor_id]
            stats["last_seen"] = datetime.now().isoformat()
            stats["packets"] += 1
            
        except queue.Full:
            logger.warning(f"Data queue full! Dropping packet from {sensor_id}")
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")

    def forward_worker(self):
        """Background thread that forwards data to the Cloud API."""
        while self.is_running:
            try:
                # 1. Check for retries first if internet is back
                if self.retry_buffer:
                    self.process_retries()

                # 2. Get next item from main queue
                data = self.data_queue.get(timeout=2)
                self.send_to_cloud(data)
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    def send_to_cloud(self, data):
        """Sends data to the /api/pipeline/analyze endpoint."""
        sensor_id = data.get("sensor_id")
        metadata = data.get("metadata", {})
        
        # Format for the Cloud API
        payload = {
            "audio_base64": data.get("audio_base64"),
            "temperature": metadata.get("temp", 20.0),
            "humidity": metadata.get("humi", 50.0),
            "wind_speed": metadata.get("wind", 5.0),
            "rainfall": metadata.get("rain", 0.0),
            "sensor_id": sensor_id,
            "region": data.get("region", "Codrii"), # Normally fetched from registry
            "latitude": data.get("lat", 47.1167),
            "longitude": data.get("lon", 28.3333)
        }

        try:
            response = requests.post(CLOUD_API_URL, json=payload, timeout=5)
            if response.status_code == 200:
                logger.info(f"Successfully forwarded packet from {sensor_id}. Latency: {response.json().get('latency_ms')}ms")
            else:
                logger.error(f"Cloud API returned error {response.status_code} for {sensor_id}")
                self.handle_failure(data)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection to Cloud API failed: {e}")
            self.handle_failure(data)

    def handle_failure(self, data):
        """Adds failed requests to the retry buffer."""
        sensor_id = data.get("sensor_id")
        if sensor_id in self.sensor_health:
            self.sensor_health[sensor_id]["errors"] += 1
        
        self.retry_buffer.append(data)
        logger.warning(f"Request for {sensor_id} added to retry buffer. Size: {len(self.retry_buffer)}")

    def process_retries(self):
        """Attempts to resend failed requests."""
        logger.info("Attempting to process retry buffer...")
        # Simple logic: try the first one, if it fails, stop retrying for now
        if not self.retry_buffer:
            return
            
        data = self.retry_buffer[0]
        try:
            # Short timeout to avoid blocking if still offline
            response = requests.get("http://google.com", timeout=2) 
            if response.status_code == 200:
                # Online! Re-send the buffered item
                self.send_to_cloud(data)
                self.retry_buffer.popleft()
        except:
            pass # Still offline

    def start(self):
        logger.info("Starting Forest Gateway...")
        threading.Thread(target=self.forward_worker, daemon=True).start()
        
        # In a real scenario, we would initialize MQTT client here
        # self.mqtt_client.on_message = self.on_mqtt_message
        # self.mqtt_client.connect("localhost", 1883)
        # self.mqtt_client.loop_forever()
        
        # For simulation, just stay alive
        try:
            while True:
                time.sleep(1)
                # Periodically log health
                if int(time.time()) % 60 == 0:
                    logger.info(f"Health Stats: {self.sensor_health}")
        except KeyboardInterrupt:
            self.is_running = False
            logger.info("Shutting down...")

if __name__ == "__main__":
    gateway = Gateway()
    gateway.start()
