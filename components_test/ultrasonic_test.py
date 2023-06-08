import RPi.GPIO as GPIO
import time

# Set GPIO pin numbering mode
GPIO.setmode(GPIO.BCM)

# Define GPIO pins
TRIG_PIN = 17
ECHO_PIN = 27

# Set up GPIO pins
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

def distance_measurement():
    # Trigger the ultrasonic sensor
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    # Wait for the ECHO pin to go high
    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()

    # Wait for the ECHO pin to go low
    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()

    # Calculate distance from the time taken
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)

    return distance

try:
    while True:
        dist = distance_measurement()
        print("Distance:", dist, "cm")
        time.sleep(1)

except KeyboardInterrupt:
    print("Measurement stopped by user")
    GPIO.cleanup()