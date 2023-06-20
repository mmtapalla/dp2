import RPi.GPIO as GPIO
import gtts
import os

# Configure GPIO pins
BUTTON_PINS = [17, 18, 27, 22, 23]

# Set up GTTS
language = 'en'  # Change to the desired language if needed
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "button_audio.mp3")

def button_callback(channel):
    text = "Button {} pressed".format(BUTTON_PINS.index(channel) + 1)
    tts = gtts.gTTS(text=text, lang=language)
    tts.save(output_file)
    os.system("mpg321 {}".format(output_file))

def setup_gpio():
    # Set GPIO mode and setup button pins
    GPIO.setmode(GPIO.BCM)
    for pin in BUTTON_PINS:
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(pin, GPIO.FALLING, callback=button_callback, bouncetime=200)

def cleanup_gpio():
    GPIO.cleanup()

def main():
    setup_gpio()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass
    cleanup_gpio()

if __name__ == '__main__':
    main()
