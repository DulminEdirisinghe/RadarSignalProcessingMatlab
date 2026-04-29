import os
import time
import base64
import paho.mqtt.client as mqtt

# ===== CONFIG =====
FOLDER_PATH = r"C:\Users\asus\Downloads\SCD_Images_224\train-set"
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "scd/images"
CHECK_INTERVAL = 1

# ==================
os.makedirs(FOLDER_PATH, exist_ok=True)

# Keep track of sent files
sent_files = set()

# MQTT setup
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

def is_image(file_name):
    return file_name.lower().endswith((".jpg", ".jpeg", ".png"))

print("Watching folder:", FOLDER_PATH)

while True:
    try:
        files = os.listdir(FOLDER_PATH)

        for file in files:
            if file not in sent_files and is_image(file):
                file_path = os.path.join(FOLDER_PATH, file)

                with open(file_path, "rb") as f:
                    image_data = f.read()

                # Encode to Base64 string
                encoded_data = base64.b64encode(image_data).decode('utf-8')

                # Send as text
                client.publish(MQTT_TOPIC, encoded_data, qos=1)

                print(f"Sent (Base64): {file}")
                sent_files.add(file)

        time.sleep(CHECK_INTERVAL)

    except Exception as e:
        print("Error:", e)
        time.sleep(2)