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
# from gtts import gTTS
from tflite_support.task import core, processor, vision

# Constants
MODEL_PATH = 'model/dp2.tflite'
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
                engine = pyttsx3.init()
                error_message = "Unable to read from the webcam. Please verify your webcam settings!"
                raise RuntimeError(error_message)
                engine.save_to_file(error_message, "audio/error_cam.mp3")
                engine.runAndWait()
                time.sleep(1)
                os.system("mpg321 -q audio/error_cam.mp3")
            image, classes = self.process_image(image, detector)
            # Perform interaction
            ProgramProper.interaction(classes, cap)
            cv2.imshow('4301 Hazard Detector', image)
            if cv2.waitKey(1) == 27: # Press 'Esc' key to exit
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
    def on_greet():
        print("Hazard detector on! Auto Mode!")
        # gTTS("Hazard detector on! Auto Mode!").save("audio/boot.mp3")
        time.sleep(1)
        os.system("mpg321 -q audio/boot.mp3")

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
                # gTTS("Manual Mode!").save("audio/manual.mp3")
                time.sleep(1)
                os.system("mpg321 -q audio/manual.mp3")
            if GPIO.input(BUTTON_PIN_3) == GPIO.LOW:
                ProgramProper.toggle_max_objects()
            if GPIO.input(BUTTON_PIN_4) == GPIO.LOW:
                ProgramProper.toggle_prob_threshold()
            if GPIO.input(BUTTON_PIN_5) == GPIO.LOW:
                ProgramProper.shutdown()
            ProgramProper.process_auto_mode(freq, current_time, cap)

        else:
            if GPIO.input(BUTTON_PIN_1) == GPIO.LOW:
                ProgramProper.AUTO_MODE = True
                print("Auto Mode!")
                # gTTS("Auto Mode!").save("audio/auto.mp3")
                time.sleep(1)
                os.system("mpg321 -q audio/auto.mp3")
            if GPIO.input(BUTTON_PIN_2) == GPIO.LOW:
                ProgramProper.process_manual_mode(freq, current_time, cap)
            if GPIO.input(BUTTON_PIN_3) == GPIO.LOW:
                ProgramProper.toggle_max_objects()
            if GPIO.input(BUTTON_PIN_4) == GPIO.LOW:
                ProgramProper.toggle_prob_threshold()
            if GPIO.input(BUTTON_PIN_5) == GPIO.LOW:
                ProgramProper.shutdown()

    @staticmethod
    def toggle_auto_interval():
        if ProgramProper.AUTO_INTERVAL == 10:
            ProgramProper.AUTO_INTERVAL = 20
            print(f"{ProgramProper.AUTO_INTERVAL}-second interval!")
            # gTTS(f"{ProgramProper.AUTO_INTERVAL}-second interval!").save(f"audio/auto_{ProgramProper.AUTO_INTERVAL}s-int.mp3")
            time.sleep(1)
            os.system(f"mpg321 -q audio/auto_{ProgramProper.AUTO_INTERVAL}s-int.mp3")
        elif ProgramProper.AUTO_INTERVAL == 20:
            ProgramProper.AUTO_INTERVAL = 30
            print(f"{ProgramProper.AUTO_INTERVAL}-second interval!")
            # gTTS(f"{ProgramProper.AUTO_INTERVAL}-second interval!").save(f"audio/auto_{ProgramProper.AUTO_INTERVAL}s-int.mp3")
            time.sleep(1)
            os.system(f"mpg321 -q audio/auto_{ProgramProper.AUTO_INTERVAL}s-int.mp3")
        else:
            ProgramProper.AUTO_INTERVAL = 10
            print(f"{ProgramProper.AUTO_INTERVAL}-second interval!")
            # gTTS(f"{ProgramProper.AUTO_INTERVAL}-second interval!").save(f"audio/auto_{ProgramProper.AUTO_INTERVAL}s-int.mp3")
            time.sleep(1)
            os.system(f"mpg321 -q audio/auto_{ProgramProper.AUTO_INTERVAL}s-int.mp3")
            
    @staticmethod
    def toggle_max_objects():
        global MAX_OBJ
        MAX_OBJ += 1
        if MAX_OBJ > 10:
            MAX_OBJ = 5
        print(f"{MAX_OBJ} max detections!")
        # gTTS(f"{MAX_OBJ} max detections!").save(f"audio/max_{MAX_OBJ}-det.mp3")
        time.sleep(1)
        os.system(f"mpg321 -q audio/max_{MAX_OBJ}-det.mp3")
        
    @staticmethod
    def toggle_prob_threshold():
        global PROB_THRESHOLD
        if PROB_THRESHOLD == 25:
            PROB_THRESHOLD = 50
            print(f"{PROB_THRESHOLD}% probability threshold!")
            # gTTS("Medium Probability!").save(f"audio/prob_{PROB_THRESHOLD}.mp3")
            time.sleep(1)
            os.system(f"mpg321 -q audio/prob_{PROB_THRESHOLD}.mp3")
        elif PROB_THRESHOLD == 50:
            PROB_THRESHOLD = 75
            print(f"{PROB_THRESHOLD}% probability threshold!")
            # gTTS("High Probability!").save(f"audio/prob_{PROB_THRESHOLD}.mp3")
            time.sleep(1)
            os.system(f"mpg321 -q audio/prob_{PROB_THRESHOLD}.mp3")
        else:
            PROB_THRESHOLD = 25
            print(f"{PROB_THRESHOLD}% probability threshold!")
            # gTTS("Low Probability!").save(f"audio/prob_{PROB_THRESHOLD}.mp3")
            time.sleep(1)
            os.system(f"mpg321 -q audio/prob_{PROB_THRESHOLD}.mp3")
        
    @staticmethod
    def shutdown():
        print("Choose an acton: [1] Shutdown or [2] Reboot")
        # gTTS("Choose an action: Press 1 to shutdown or 2 to reboot.").save("audio/off-prompt.mp3")
        time.sleep(1)
        os.system("mpg321 -q audio/off-prompt.mp3")

        while True:
            if (GPIO.input(BUTTON_PIN_1) == GPIO.LOW):
                print("Shutting down!")
                # gTTS("Shutting down!").save("audio/shut.mp3")
                time.sleep(1)
                os.system("mpg321 -q audio/shut.mp3")
                subprocess.run(["sudo", "shutdown", "-h", "now"])
            elif (GPIO.input(BUTTON_PIN_2) == GPIO.LOW):
                print("Rebooting!")
                # gTTS("Rebooting!").save("audio/shut_reboot.mp3")
                time.sleep(1)
                os.system("mpg321 -q audio/shut_reboot.mp3")
                subprocess.run(["sudo", "reboot", "-h", "now"])
            elif (
                GPIO.input(BUTTON_PIN_3) == GPIO.LOW
                or GPIO.input(BUTTON_PIN_4) == GPIO.LOW
                or GPIO.input(BUTTON_PIN_5) == GPIO.LOW
            ):
                print("Shutdown cancelled!")
                # gTTS("Shutdown cancelled!").save("audio/shut_cancelled.mp3")
                time.sleep(1)
                os.system("mpg321 -q audio/shut_cancelled.mp3")
                break
        
    @staticmethod
    def process_auto_mode(freq, current_time, cap):
        if bool(freq):
            if time.time() - ProgramProper.AUTO_COUNTER >= ProgramProper.AUTO_INTERVAL and bool(freq):
                ProgramProper.AUTO_COUNTER = time.time()
                image_name = re.sub(r'[/:_]', '', current_time)
                image_path = f"image/{image_name}.jpg"
                voice = ", and ".join([f"{count} {item}" for item, count in freq.items()])

                # Initialize pyttsx3 engine
                engine = pyttsx3.init()
                engine.save_to_file(voice, "audio/detection.mp3")
                engine.runAndWait()

                time.sleep(1)
                os.system("mpg321 -q audio/detection.mp3")
                freq['time'] = current_time
                print(freq)
                ret, frame = cap.read()
                if ret:
                    # cv2.imwrite(image_path, frame)
                    # print(f"Image saved: {image_path}")
                    pass
                else:
                    # print("Failed to capture image")
                    # gTTS("Failed to capture image").save("audio/capture_failed.mp3")
                    # time.sleep(1)
                    # os.system("mpg321 -q audio/capture_failed.mp3")
                    pass

    @staticmethod
    def process_manual_mode(freq, current_time, cap):
        if bool(freq):
            image_name = re.sub(r'[/:_]', '', current_time)
            image_path = f"image/{image_name}.jpg"
            voice = ", and ".join([f"{count} {item}" for item, count in freq.items()])

            # Initialize pyttsx3 engine
            engine = pyttsx3.init()
            engine.save_to_file(voice, "audio/detection.mp3")
            engine.runAndWait()

            time.sleep(1)
            os.system("mpg321 -q audio/detection.mp3")
            freq['time'] = current_time
            print(freq)
            ret, frame = cap.read()
            if ret:
                # cv2.imwrite(image_path, frame)
                # print(f"Image saved: {image_path}")
                pass
            else:
                # print("Failed to capture image")
                # gTTS("Failed to capture image").save("audio/error_capture.mp3")
                # time.sleep(1)
                # os.system("mpg321 -q audio/error_capture.mp3")
                pass
        else:
            print("No detections!")
            # gTTS("No detections!").save("audio/detection_none.mp3")
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
    program = ProgramProper()
    program.on_greet()
    program.main()