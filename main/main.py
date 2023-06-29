import argparse
import cv2
import numpy as np
import os
import psutil
import pyttsx3
import re
import RPi.GPIO as GPIO
import subprocess
import time
from collections import Counter
from datetime import datetime
from tflite_support.task import core, processor, vision

# Constants
MODEL_PATH = 'model/MobileNetSSDv2.tflite'
PROB_THRESHOLD = 50
MAX_OBJ = 5
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (52, 29, 197)

# GPIO Constants
""" Auto Mode, Manual Mode, Max Detection Quantity, Propabilty Threshold, Shutdown"""
BUTTON_PINS = [17, 18, 27, 22, 23]

greet = True

class ObjectDetector:
    def __init__(self, model_path, margin, row_size, font_size, font_thickness, text_color):
        self.model_path = model_path
        self.margin = margin
        self.row_size = row_size
        self.font_size = font_size
        self.font_thickness = font_thickness
        self.text_color = text_color

    def visualize_detections(self, image, detection_result):
        """Visualize object detections on the image."""
        classes = []
        for detection in detection_result.detections:
            category = detection.categories[0]
            probability = round(category.score * 100)
            if probability >= PROB_THRESHOLD:
                bbox = detection.bounding_box
                start_point, end_point = (bbox.origin_x, bbox.origin_y), (
                    bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                cv2.rectangle(image, start_point, end_point, self.text_color, 3)
                class_name = category.category_name
                classes.append(class_name)
                result_text = f"{class_name} ({probability}%)"
                text_location = (self.margin + bbox.origin_x, self.margin + self.row_size + bbox.origin_y)
                cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, self.font_size, self.text_color,
                            self.font_thickness)
        return image, classes

    def process_image(self, image, detector):
        """Process the image by performing object detection and visualization."""
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = vision.TensorImage.create_from_array(rgb_image)
        detection_result = detector.detect(input_tensor)
        image, classes = self.visualize_detections(image, detection_result)
        return image, classes

    def run_inference(self, camera_id, width, height, num_threads, enable_edgetpu):
        global greet
        """Run the object detection inference loop."""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Initialize the object detector
        base_options = core.BaseOptions(file_name=self.model_path, use_coral=enable_edgetpu, num_threads=num_threads)
        detection_options = processor.DetectionOptions(max_results=MAX_OBJ, score_threshold=0.3)
        options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
        detector = vision.ObjectDetector.create_from_options(options)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                raise RuntimeError("Unable to read from the webcam. Please verify your webcam settings!")

            image, classes = self.process_image(image, detector)
            ProgramProper.interaction(classes, cap)
            cv2.imshow('4301 Object Detector', image)
            if bool(greet):
                ProgramProper.on_greet()
            if cv2.waitKey(1) == 27:  # Press 'Esc' key to exit
                break
            if MAX_OBJ != detection_options.max_results:
                detection_options = processor.DetectionOptions(max_results=MAX_OBJ, score_threshold=0.3)
                options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
                detector = vision.ObjectDetector.create_from_options(options)
        cap.release()
        cv2.destroyAllWindows()


class ProgramProper:
    AUTO_MODE = True
    AUTO_COUNTER = time.time()
    AUTO_INTERVAL = 10

    @staticmethod
    def setup_gpio():
        """Set up GPIO pins and event detection."""
        GPIO.setmode(GPIO.BCM)
        for pin in BUTTON_PINS:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    @staticmethod
    def on_greet():
        global greet
        greet = False
        print("+++ 4301 Object Detector +++\nAUTO Mode!")
        time.sleep(1)
        os.system("mpg321 -q audio/model_object.mp3")

    @staticmethod
    def interaction(classes, cap):
        freq = dict(Counter(classes))
        current_time = datetime.now().strftime("%Y/%m/%d_%H:%M:%S")

        if ProgramProper.AUTO_MODE:
            if GPIO.input(BUTTON_PINS[0]) == GPIO.LOW:
                ProgramProper.toggle_auto_interval()
            if GPIO.input(BUTTON_PINS[1]) == GPIO.LOW:
                ProgramProper.AUTO_MODE = False
                print("MANUAL Mode!")
                time.sleep(1)
                os.system("mpg321 -q audio/manual.mp3")
            if GPIO.input(BUTTON_PINS[2]) == GPIO.LOW:
                ProgramProper.toggle_max_obj_qty()
            if GPIO.input(BUTTON_PINS[3]) == GPIO.LOW:
                ProgramProper.toggle_prob_threshold()
            if GPIO.input(BUTTON_PINS[4]) == GPIO.LOW:
                ProgramProper.shutdown()
            ProgramProper.auto_mode(freq, current_time, cap)

        else:
            if GPIO.input(BUTTON_PINS[0]) == GPIO.LOW:
                ProgramProper.AUTO_MODE = True
                print("AUTO Mode!")
                time.sleep(1)
                os.system("mpg321 -q audio/auto.mp3")
            if GPIO.input(BUTTON_PINS[1]) == GPIO.LOW:
                ProgramProper.manual_mode(freq, current_time, cap)
            if GPIO.input(BUTTON_PINS[2]) == GPIO.LOW:
                ProgramProper.toggle_max_obj_qty()
            if GPIO.input(BUTTON_PINS[3]) == GPIO.LOW:
                ProgramProper.toggle_prob_threshold()
            if GPIO.input(BUTTON_PINS[4]) == GPIO.LOW:
                ProgramProper.shutdown()

    @staticmethod
    def toggle_auto_interval():
        auto_intervals = [10, 20, 30]
        ProgramProper.AUTO_INTERVAL = auto_intervals[(auto_intervals.index(ProgramProper.AUTO_INTERVAL) + 1) % len(auto_intervals)]
        print(f"{ProgramProper.AUTO_INTERVAL}-second interval!")
        time.sleep(1)
        os.system(f"mpg321 -q audio/auto_{ProgramProper.AUTO_INTERVAL}s-int.mp3")

    @staticmethod
    def toggle_max_obj_qty():
        global MAX_OBJ
        MAX_OBJ = 5 if MAX_OBJ >= 10 else MAX_OBJ + 1
        print(f"{MAX_OBJ} max detections!")
        time.sleep(1)
        os.system(f"mpg321 -q audio/max_{MAX_OBJ}-det.mp3")

    @staticmethod
    def toggle_prob_threshold():
        global PROB_THRESHOLD
        prob_thresholds = [25, 50, 75]
        PROB_THRESHOLD = prob_thresholds[(prob_thresholds.index(PROB_THRESHOLD) + 1) % len(prob_thresholds)]
        print(f"{PROB_THRESHOLD}% probability threshold!")
        time.sleep(1)
        os.system(f"mpg321 -q audio/prob_{PROB_THRESHOLD}.mp3")

    @staticmethod
    def shutdown():
        print("Choose an action: [1] Shutdown or [2] Reboot")
        time.sleep(1)
        os.system("mpg321 -q audio/off-prompt.mp3")

        while True:
            if GPIO.input(BUTTON_PINS[0]) == GPIO.LOW:
                print("Shutting down!")
                time.sleep(1)
                os.system("mpg321 -q audio/shut.mp3")
                subprocess.run(["sudo", "shutdown", "-h", "now"])
            elif GPIO.input(BUTTON_PINS[1]) == GPIO.LOW:
                print("Rebooting!")
                time.sleep(1)
                os.system("mpg321 -q audio/shut_reboot.mp3")
                subprocess.run(["sudo", "reboot", "-h", "now"])
            elif any(GPIO.input(pin) == GPIO.LOW for pin in BUTTON_PINS[2:]):
                print("Shutdown cancelled!")
                time.sleep(1)
                os.system("mpg321 -q audio/shut_cancelled.mp3")
                break

    @staticmethod
    def auto_mode(freq, current_time, cap):
        if bool(freq) and time.time() - ProgramProper.AUTO_COUNTER >= ProgramProper.AUTO_INTERVAL:
            ProgramProper.AUTO_COUNTER = time.time()
            voice = ", and ".join([f"{count} {item}" for item, count in freq.items()])

            engine = pyttsx3.init()
            engine.save_to_file(voice, "audio/detection.mp3")
            engine.runAndWait()

            time.sleep(1)
            os.system("mpg321 -q audio/detection.mp3")
            freq['time'] = current_time
            print(f"A: {freq}")
            ret, frame = cap.read()

    @staticmethod
    def manual_mode(freq, current_time, cap):
        if bool(freq):
            voice = ", and ".join([f"{count} {item}" for item, count in freq.items()])

            engine = pyttsx3.init()
            engine.save_to_file(voice, "audio/detection.mp3")
            engine.runAndWait()

            time.sleep(1)
            os.system("mpg321 -q audio/detection.mp3")
            freq['time'] = current_time
            print(f"M: {freq}")
            ret, frame = cap.read()
        else:
            print("No detections!")
            time.sleep(1)
            os.system("mpg321 -q audio/detection_none.mp3")

    @staticmethod
    def main():
        """Main function."""
        args = ProgramProper.parse_arguments()
        ProgramProper.setup_gpio()
        detector = ObjectDetector(args.model, MARGIN, ROW_SIZE, FONT_SIZE, FONT_THICKNESS, TEXT_COLOR)
        detector.run_inference(args.cameraId, args.frameWidth, args.frameHeight, args.numThreads, args.enableEdgeTPU)

    @staticmethod
    def parse_arguments():
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--model', help='Path of the object detection model.', default=MODEL_PATH)
        parser.add_argument('--cameraId', help='Id of camera.', type=int, default=0)
        parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', type=int, default=640)
        parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', type=int, default=480)
        parser.add_argument('--numThreads', help='Number of CPU threads to run the model.', type=int, default=4)
        parser.add_argument('--enableEdgeTPU', help='Whether to run the model on EdgeTPU.', action='store_true',
                            default=False)
        return parser.parse_args()

if __name__ == '__main__':
    ProgramProper.main()