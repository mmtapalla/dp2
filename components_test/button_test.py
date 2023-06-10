import RPi.GPIO as GPIO
from gtts import gTTS
import subprocess

# Configuration for button pins and corresponding modes
button_config = {17: "Auto mode", 18: "Manual mode"}

def setup_buttons():
    # Set the GPIO mode and specify the pin numbers for the buttons
    GPIO.setmode(GPIO.BCM)

    # Set up the GPIO pins as input with a pull-up resistor
    for pin in button_config:
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def button_callback(channel):
    mode = button_config.get(channel)
    if mode:
        # Generate the audio using gTTS
        tts = gTTS(mode)
        filename = f"{mode.lower().replace(' ', '_')}.mp3"
        tts.save(filename)

        # Print the mode to the terminal
        print(f"{mode}")

        # Play the audio using mpg321
        subprocess.run(["mpg321", "-q", filename], check=True)

def main():
    setup_buttons()

    # Add event detection for both buttons, only on the falling edge
    for pin in button_config:
        GPIO.add_event_detect(pin, GPIO.FALLING, callback=button_callback, bouncetime=200)

    # Keep the script running until interrupted
    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up the GPIO settings
        GPIO.cleanup()

if __name__ == "__main__":
    main()
