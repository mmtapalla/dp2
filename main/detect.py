import argparse
import sys
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from gtts import gTTS
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import os


# Configuration variables
_MODEL_PATH = 'efficientdet_lite0.tflite'
_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (52, 29, 197)  # Raspberry Pi Red

# GPIO Push Buttons
BUTTON_PIN_1 = 17
BUTTON_PIN_2 = 18


def speak(voice):
    """Convert text to speech using gTTS and play the audio."""
    tts = gTTS(text=voice)
    tts.save("detect.mp3")
    os.system("mpg321 detect.mp3")


def count_freq(li, button_pin):
    """Count the frequency of items in a list and speak the results."""
    freq = {}
    for item in li:
        freq[item] = li.count(item)

    if GPIO.input(button_pin) == GPIO.LOW:
        voice = ""
        for item, count in freq.items():
            if voice:
                voice += ", "
            voice += f"{count} {item}"
        voice = voice.replace(", ", ", and ")
        speak(voice)
        print(freq)


def visualize_detections(image, detection_result, button_pin, margin, row_size, font_size, font_thickness, text_color):
    """Draw bounding boxes on the image based on the detection results."""
    classes = []
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, text_color, 3)

        category = detection.categories[0]
        class_name = category.category_name
        classes.append(class_name)
        probability = round(category.score, 2)
        result_text = f"{class_name} ({probability})"
        text_location = (margin + bbox.origin_x, margin + row_size + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

    count_freq(classes, button_pin)

    return image


def process_image(image, detector, button_pin, margin, row_size, font_size, font_thickness, text_color):
    """Process a single image frame: flip, convert colors, run inference, and visualize detections."""
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = vision.TensorImage.create_from_array(rgb_image)
    detection_result = detector.detect(input_tensor)
    image = visualize_detections(image, detection_result, button_pin, margin, row_size, font_size, font_thickness, text_color)

    return image


def run_inference(model_path, camera_id, width, height, num_threads, enable_edgetpu):
    """Run the object detection model on live video input."""
    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Initialize the object detection model
    base_options = core.BaseOptions(file_name=model_path, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            raise ValueError('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        counter += 1
        image = process_image(image, detector, BUTTON_PIN_2, _MARGIN, _ROW_SIZE, _FONT_SIZE, _FONT_THICKNESS, _TEXT_COLOR)

        # Calculate the FPS
        if counter % 10 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = f"FPS = {fps:.1f}"
        text_location = (24, 20)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

        # Display the image and exit if ESC key is pressed
        cv2.imshow('object_detector', image)
        if cv2.waitKey(1) == 27:
            break

    # Release video capture object and destroy windows
    cap.release()
    cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Path of the object detection model.', default=_MODEL_PATH)
    parser.add_argument('--cameraId', help='Id of camera.', type=int, default=0)
    parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', type=int, default=640)
    parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', type=int, default=480)
    parser.add_argument('--numThreads', help='Number of CPU threads to run the model.', type=int, default=4)
    parser.add_argument('--enableEdgeTPU', help='Whether to run the model on EdgeTPU.', action='store_true', default=False)
    return parser.parse_args()


def button1_callback(channel):
    """Button 1 callback function."""
    print("Button 1 pressed")


def button2_callback(channel):
    """Button 2 callback function."""
    print("Button 2 pressed")


def setup_gpio():
    """Set up GPIO pins and event detection for buttons."""
    # Initialize GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN_1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(BUTTON_PIN_2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(BUTTON_PIN_1, GPIO.FALLING, callback=button1_callback, bouncetime=200)
    GPIO.add_event_detect(BUTTON_PIN_2, GPIO.FALLING, callback=button2_callback, bouncetime=200)


def main():
    args = parse_arguments()
    setup_gpio()
    run_inference(args.model, args.cameraId, args.frameWidth, args.frameHeight, args.numThreads, args.enableEdgeTPU)


if __name__ == '__main__':
    main()