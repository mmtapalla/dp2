import RPi.GPIO as GPIO
import time

# Set the GPIO mode to BCM
GPIO.setmode(GPIO.BCM)

# Define the GPIO pin number
button_pin = 0

# Set up the GPIO pin for the button
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

try:
    while True:
        # Read the state of the button
        button_state = GPIO.input(button_pin)

        if button_state == GPIO.LOW:
            print("Button pressed!")
        
        # Add a small delay to debounce the button
        time.sleep(0.1)

except KeyboardInterrupt:
    # Clean up GPIO settings on keyboard interrupt
    GPIO.cleanup()