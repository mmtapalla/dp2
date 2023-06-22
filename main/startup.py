import time
import os
import subprocess
import RPi.GPIO as GPIO
from gtts import gTTS

BUTTON_PIN_1 = 17 # Button 1
BUTTON_PIN_2 = 18 # Button 2
BUTTON_PIN_3 = 27
BUTTON_PIN_4 = 22
BUTTON_PIN_5 = 23

print("Choose a model: [1] Hazard Detector or [2] Object Detector")
# gTTS("Choose a model: Press 1 for Hazard Detector or 2 for Object Detector.").save("audio/model_prompt.mp3")
time.sleep(.5)
os.system("mpg321 audio/model_prompt.mp3")

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN_1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_PIN_2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_PIN_3, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_PIN_4, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_PIN_5, GPIO.IN, pull_up_down=GPIO.PUD_UP)

while True:
    if GPIO.input(BUTTON_PIN_1) == GPIO.LOW:
        print("Loading Hazard Detector!")
        # gTTS("Loading Hazard Detector!").save("audio/model_hazard.mp3")
        time.sleep(1)
        os.system("mpg321 audio/model_hazard.mp3")
        subprocess.run(["python", "main.py"])
        break
    elif GPIO.input(BUTTON_PIN_2) == GPIO.LOW:
        print("Loading Object Detector!")
        # gTTS("Loading Object Detector!").save("audio/model_object.mp3")
        time.sleep(1)
        os.system("mpg321 audio/model_object.mp3")
        subprocess.run(["python", "main-alt.py"])
        break
    elif (
        GPIO.input(BUTTON_PIN_3) == GPIO.LOW
        or GPIO.input(BUTTON_PIN_4) == GPIO.LOW
        or GPIO.input(BUTTON_PIN_5) == GPIO.LOW
    ):
        time.sleep(1)
        os.system("mpg321 audio/model_prompt.mp3")
        

GPIO.cleanup()