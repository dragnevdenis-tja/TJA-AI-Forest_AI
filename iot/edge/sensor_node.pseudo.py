"""
Forest Audio AI - Sensor Node Pseudo-code (ESP32-S3)
This represents the firmware logic for a single sensor node.
"""

import time
import hardware_i2s as i2s
import hardware_sensors as sensors
import network_mqtt as mqtt
import storage_sd as sd

# Configuration
SENSOR_ID = "SN-COD-202"
GATEWAY_IP = "192.168.1.50"
ENERGY_THRESHOLD = 0.005 # Skip transmission if sound is too quiet
CHUNK_DURATION_SEC = 5
SAMPLE_RATE = 32000

def initialize():
    i2s.init(sample_rate=SAMPLE_RATE, bits_per_sample=16)
    sensors.init()
    sd.init()
    mqtt.connect(GATEWAY_IP, client_id=SENSOR_ID)

def capture_and_process():
    while True:
        try:
            # 1. Capture 5 seconds of audio
            print(f"[{SENSOR_ID}] Recording...")
            audio_buffer = i2s.record(duration=CHUNK_DURATION_SEC)
            
            # 2. Capture environmental metadata
            env_data = {
                "temp": sensors.get_temperature(),
                "humi": sensors.get_humidity(),
                "pres": sensors.get_pressure()
            }
            
            # 3. Energy thresholding (Edge Filtering)
            rms_energy = calculate_rms(audio_buffer)
            if rms_energy < ENERGY_THRESHOLD:
                print("[-] Audio below energy threshold. Skipping transmission.")
                continue
            
            # 4. Prepare payload
            payload = {
                "sensor_id": SENSOR_ID,
                "metadata": env_data,
                "audio_base64": base64_encode(audio_buffer),
                "timestamp": time.now_iso()
            }
            
            # 5. Transmission logic
            if mqtt.is_connected():
                # Publish to gateway
                success = mqtt.publish(f"forest/audio/{SENSOR_ID}", payload)
                if not success:
                    queue_for_offline_sync(payload)
            else:
                print("[!] Connection lost. Saving to SD card.")
                queue_for_offline_sync(payload)
                attempt_reconnect()
                
        except Exception as e:
            print(f"[ERROR] Node failure: {e}")
            time.sleep(10)

def queue_for_offline_sync(payload):
    # Append payload to a rolling buffer on the SD card
    filename = f"/sd/queue_{time.timestamp()}.json"
    sd.write_file(filename, payload)

def attempt_reconnect():
    # Exponential backoff for MQTT reconnection
    retry_delay = 1
    while not mqtt.is_connected():
        print(f"[*] Attempting reconnect in {retry_delay}s...")
        time.sleep(retry_delay)
        if mqtt.connect(GATEWAY_IP):
            print("[+] Reconnected to Gateway.")
            sync_offline_data() # Slowly drain SD card queue
            break
        retry_delay = min(retry_delay * 2, 300) # Max 5 mins

def calculate_rms(audio_buffer):
    # Square root of the mean of squares (standard energy metric)
    sum_squares = sum(sample**2 for sample in audio_buffer)
    return (sum_squares / len(audio_buffer))**0.5

if __name__ == "__main__":
    initialize()
    capture_and_process()
