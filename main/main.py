import argparse
import cv2
import numpy as np
import os
import RPi.GPIO as GPIO
import time
from collections import Counter
from gtts import gTTS
from tflite_support.task import core, processor, vision

MODEL_PATH = 'efficientdet_lite0.tflite'
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (52, 29, 197)

def visualize_detections(image, detection_result, margin, row_size, font_size, font_thickness, text_color):
    """Visualize object detections on the image."""
    classes = []
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point, end_point = (bbox.origin_x, bbox.origin_y), (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(image, start_point, end_point, text_color, 3)
        category = detection.categories[0]
        class_name = category.category_name
        classes.append(class_name)
        probability = round(category.score, 2)
        result_text = f"{class_name} ({probability})"
        text_location = (margin + bbox.origin_x, margin + row_size + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)
    count_freq(classes)
    return image

def process_image(image, detector, margin, row_size, font_size, font_thickness, text_color):
    """Process the image by performing object detection and visualization."""
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = vision.TensorImage.create_from_array(rgb_image)
    detection_result = detector.detect(input_tensor)
    image = visualize_detections(image, detection_result, margin, row_size, font_size, font_thickness, text_color)
    return image

def run_inference(model_path, camera_id, width, height, num_threads, enable_edgetpu):
    """Run the object detection inference loop."""
    counter, fps = 0, 0
    start_time = time.time()

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Initialize the object detector
    base_options = core.BaseOptions(file_name=model_path, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(max_results=5, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            raise RuntimeError('Unable to read from the webcam. Please verify your webcam settings.')

        counter += 1
        image = process_image(image, detector, MARGIN, ROW_SIZE, FONT_SIZE, FONT_THICKNESS, TEXT_COLOR)

        if counter % 10 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = time.time()

        fps_text = f"FPS = {fps:.1f}"
        text_location = (24, 20)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        cv2.imshow('4301 Hazard Detector', image)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Path of the object detection model.', default=MODEL_PATH)
    parser.add_argument('--cameraId', help='Id of camera.', type=int, default=0)
    parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', type=int, default=640)
    parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', type=int, default=480)
    parser.add_argument('--numThreads', help='Number of CPU threads to run the model.', type=int, default=4)
    parser.add_argument('--enableEdgeTPU', help='Whether to run the model on EdgeTPU.', action='store_true', default=False)
    return parser.parse_args()

BUTTON_PIN_1 = 17
BUTTON_PIN_2 = 18

def setup_gpio():
    """Set up GPIO pins and event detection."""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN_1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(BUTTON_PIN_2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def speak(voice):
    """Convert text to speech and play the audio."""
    tts = gTTS(text=voice)
    tts.save("detect.mp3")
    os.system("mpg321 detect.mp3")

def count_freq(classes):
    """Count the frequency of items in a list and output the result."""
    freq = Counter(classes)
    if GPIO.input(BUTTON_PIN_2) == GPIO.LOW:
        voice = ", and ".join([f"{count} {item}" for item, count in freq.items()])
        speak(voice)
        print(freq)

def main():
    """Main function."""
    args = parse_arguments()
    setup_gpio()
    run_inference(args.model, args.cameraId, args.frameWidth, args.frameHeight, args.numThreads, args.enableEdgeTPU)

if __name__ == '__main__':
    main()