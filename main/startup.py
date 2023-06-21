import time
import os
import subprocess
import RPi.GPIO as GPIO
from gtts import gTTS

BUTTON_PIN_1 = 17 
BUTTON_PIN_2 = 18

print("Choose a model: [1] Hazard Detector or [2] Object Detector")
# tts = gTTS("Choose a model: Press 1 for Hazard Detector or 2 for Object Detector.")
# tts.save("audio/model_prompt.mp3")
time.sleep(1)
os.system("mpg321 audio/model_prompt.mp3")

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN_1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_PIN_2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

while True:
    if GPIO.input(BUTTON_PIN_1) == GPIO.LOW:
        print("Loading Hazard Detector!")
        # tts = gTTS("Loading Hazard Detector!")
        # tts.save("audio/model_hazard.mp3")
        time.sleep(1)
        os.system("mpg321 audio/model_hazard.mp3")
        subprocess.run(["python", "main.py"])
        break
    elif GPIO.input(BUTTON_PIN_2) == GPIO.LOW:
        print("Loading Object Detector!")
        # tts = gTTS("Loading Object Detector!")
        # tts.save("audio/model_object.mp3")
        time.sleep(1)
        os.system("mpg321 audio/model_object.mp3")
        subprocess.run(["python", "main-alt.py"])
        break

GPIO.cleanup()