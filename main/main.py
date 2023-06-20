import argparse
import cv2
import numpy as np
import os
import psutil
import re
import RPi.GPIO as GPIO
import subprocess
import time
from collections import Counter
from datetime import datetime
from gtts import gTTS
from tflite_support.task import core, processor, vision

# Constants
MODEL_PATH = 'model/efficientdet_lite0.tflite'
PROB_THRESHOLD = 25
MAX_OBJ = 5
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (52, 29, 197)

# GPIO Constants
BUTTON_PIN_1 = 17 # Auto Mode
BUTTON_PIN_2 = 18 # Manual Mode
BUTTON_PIN_3 = 27 # Max Detection Quantity
BUTTON_PIN_4 = 22 # Probability Threshold
BUTTON_PIN_5 = 23 # Off Switch


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
                raise RuntimeError('Unable to read from the webcam. Please verify your webcam settings!')
                tts = gTTS("Unable to read from the webcam. Please verify your webcam settings!")
                tts.save("audio/cam_error.mp3")
                time.sleep(1)
                os.system("mpg321 audio/cam_error.mp3")
            image, classes = self.process_image(image, detector)
            # Perform interaction
            ProgramProper.interaction(classes, cap)
            cv2.imshow('4301 Hazards Detector', image)
            if cv2.waitKey(1) == 27:
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
        GPIO.setup(BUTTON_PIN_1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(BUTTON_PIN_2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(BUTTON_PIN_3, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(BUTTON_PIN_4, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(BUTTON_PIN_5, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    @staticmethod
    def interaction(classes, cap):
        freq = dict(Counter(classes))
        current_time = datetime.now().strftime("%Y/%m/%d_%H:%M:%S")

        if ProgramProper.AUTO_MODE:
            if GPIO.input(BUTTON_PIN_1) == GPIO.LOW:
                ProgramProper.toggle_auto_interval()
            if GPIO.input(BUTTON_PIN_2) == GPIO.LOW:
                ProgramProper.AUTO_MODE = False
                print("Manual Mode!")
                # tts = gTTS("Manual Mode!")
                # tts.save("audio/manual_mode.mp3")
                time.sleep(1)
                os.system("mpg321 audio/manual_mode.mp3")
            if GPIO.input(BUTTON_PIN_3) == GPIO.LOW:
                ProgramProper.toggle_max_objects()
            if GPIO.input(BUTTON_PIN_4) == GPIO.LOW:
                ProgramProper.toggle_PROB_THRESHOLD()
            if GPIO.input(BUTTON_PIN_5) == GPIO.LOW:
                ProgramProper.off_switch()
            ProgramProper.process_auto_mode(freq, current_time, cap)

        else:
            if GPIO.input(BUTTON_PIN_1) == GPIO.LOW:
                ProgramProper.AUTO_MODE = True
                print("Auto Mode!")
                # tts = gTTS("Auto Mode!")
                # tts.save("audio/auto_mode.mp3")
                time.sleep(1)
                os.system("mpg321 audio/auto_mode.mp3")
            if GPIO.input(BUTTON_PIN_2) == GPIO.LOW:
                ProgramProper.process_manual_mode(freq, current_time, cap)
            if GPIO.input(BUTTON_PIN_3) == GPIO.LOW:
                ProgramProper.toggle_max_objects()
            if GPIO.input(BUTTON_PIN_4) == GPIO.LOW:
                ProgramProper.toggle_PROB_THRESHOLD()
            if GPIO.input(BUTTON_PIN_5) == GPIO.LOW:
                ProgramProper.off_switch()

    @staticmethod
    def toggle_auto_interval():
        if ProgramProper.AUTO_INTERVAL == 10:
            ProgramProper.AUTO_INTERVAL = 20
            print("20-second interval!")
            # tts = gTTS("20-second interval!")
            # tts.save("mpg321 audio/auto_20sec.mp3")
            time.sleep(1)
            os.system("mpg321 audio/auto_20sec.mp3")
        elif ProgramProper.AUTO_INTERVAL == 20:
            ProgramProper.AUTO_INTERVAL = 30
            print("30-second interval!")
            # tts = gTTS("30-second interval!")
            # tts.save("mpg321 audio/auto_30sec.mp3")
            time.sleep(1)
            os.system("mpg321 audio/auto_30sec.mp3")
        else:
            ProgramProper.AUTO_INTERVAL = 10
            print("10-second interval!")
            # tts = gTTS("10-second interval!")
            # tts.save("mpg321 audio/auto_10sec.mp3")
            time.sleep(1)
            os.system("mpg321 audio/auto_10sec.mp3")
            
    @staticmethod
    def toggle_max_objects():
        global MAX_OBJ
        MAX_OBJ += 1
        if MAX_OBJ > 10:
            MAX_OBJ = 5
        print(f"{MAX_OBJ} Max Detections!")
        # tts = gTTS(f"{MAX_OBJ} Max Detections!")
        # tts.save(f"audio/detect_{MAX_OBJ}-max.mp3")
        time.sleep(1)
        os.system(f"mpg321 audio/detect_{MAX_OBJ}-max.mp3")
        
    @staticmethod
    def toggle_PROB_THRESHOLD():
        global PROB_THRESHOLD
        if PROB_THRESHOLD == 25:
            PROB_THRESHOLD = 50
            print(f"{PROB_THRESHOLD}% probability threshold!")
            # tts = gTTS("Medium Probability!")
            # tts.save("audio/prob_50.mp3")
            time.sleep(1)
            os.system("mpg321 audio/prob_50.mp3")
        elif PROB_THRESHOLD == 50:
            PROB_THRESHOLD = 75
            print(f"{PROB_THRESHOLD}% probability threshold!")
            # tts = gTTS("High Probability!")
            # tts.save("audio/prob_75.mp3")
            time.sleep(1)
            os.system("mpg321 audio/prob_75.mp3")
        else:
            PROB_THRESHOLD = 25
            print(f"{PROB_THRESHOLD}% probability threshold!")
            # tts = gTTS("Low Probability!")
            # tts.save("audio/prob_25.mp3")
            time.sleep(1)
            os.system("mpg321 audio/prob_25.mp3")
        
    @staticmethod
    def off_switch():
        print("Warning! Do you want to switch off the device?")
        # tts = gTTS("Warning! Do you want to switch off the device?")
        # tts.save("audio/switch_off-warning.mp3")
        time.sleep(1)
        os.system("mpg321 audio/switch_off-warning.mp3")

        while True:
            if GPIO.input(BUTTON_PIN_5) == GPIO.LOW:
                print("Switching Off!")
                # tts = gTTS("Switching Off!")
                # tts.save("audio/switch_off.mp3")
                time.sleep(1)
                os.system("mpg321 audio/switch_off.mp3")
                subprocess.run(["sudo", "shutdown", "-h", "now"])
            elif (
                GPIO.input(BUTTON_PIN_1) == GPIO.LOW
                or GPIO.input(BUTTON_PIN_2) == GPIO.LOW
                or GPIO.input(BUTTON_PIN_3) == GPIO.LOW
                or GPIO.input(BUTTON_PIN_4) == GPIO.LOW
            ):
                print("Switch off cancelled!")
                # tts = gTTS("Switch off cancelled!")
                # tts.save("audio/switch_off-cancelled.mp3")
                time.sleep(1)
                os.system("mpg321 audio/switch_off-cancelled.mp3")
                break
        
    @staticmethod
    def process_auto_mode(freq, current_time, cap):
        if bool(freq):
            if time.time() - ProgramProper.AUTO_COUNTER >= ProgramProper.AUTO_INTERVAL and bool(freq):
                ProgramProper.AUTO_COUNTER = time.time()
                image_name = re.sub(r'[/:_]', '', current_time)
                image_path = f"image/{image_name}.jpg"
                voice = ", and ".join([f"{count} {item}" for item, count in freq.items()])
                tts = gTTS(text=voice)
                tts.save("audio/detect.mp3")
                time.sleep(1)
                os.system("mpg321 audio/detect.mp3")
                freq['time'] = current_time
                print(freq)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(image_path, frame)
                    print(f"Image saved: {image_path}")
                else:
                    print("Failed to capture image")
                    tts = gTTS("Failed to capture image")
                    tts.save("audio/capture_failed.mp3")
                    time.sleep(1)
                    os.system("mpg321 audio/capture_failed.mp3")

    @staticmethod
    def process_manual_mode(freq, current_time, cap):
        if bool(freq):
            image_name = re.sub(r'[/:_]', '', current_time)
            image_path = f"image/{image_name}.jpg"
            voice = ", and ".join([f"{count} {item}" for item, count in freq.items()])
            tts = gTTS(text=voice)
            tts.save("audio/detect.mp3")
            time.sleep(1)
            os.system("mpg321 audio/detect.mp3")
            freq['time'] = current_time
            print(freq)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(image_path, frame)
                print(f"Image saved: {image_path}")
            else:
                print("Failed to capture image")
                tts = gTTS("Failed to capture image")
                tts.save("audio/capture_failed.mp3")
                time.sleep(1)
                os.system("mpg321 audio/capture_failed.mp3")
        else:
            time.sleep(1)
            os.system("mpg321 audio/manual_no-detect.mp3")

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
    program = ProgramProper()
    program.main()