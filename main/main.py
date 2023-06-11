import argparse
import cv2
import numpy as np
import os
import RPi.GPIO as GPIO
import time
from collections import Counter
from gtts import gTTS
from tflite_support.task import core, processor, vision


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
            bbox = detection.bounding_box
            start_point, end_point = (bbox.origin_x, bbox.origin_y), (
                bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
            cv2.rectangle(image, start_point, end_point, self.text_color, 3)
            category = detection.categories[0]
            class_name = category.category_name
            classes.append(class_name)
            probability = round(category.score, 2)
            result_text = f"{class_name} ({probability})"
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
        detection_options = processor.DetectionOptions(max_results=5, score_threshold=0.3)
        options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
        detector = vision.ObjectDetector.create_from_options(options)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                raise RuntimeError('Unable to read from the webcam. Please verify your webcam settings.')

            image, classes = self.process_image(image, detector)

            # Perform interaction
            ProgramProper.interaction(classes)

            cv2.imshow('4301 Hazard Detector', image)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


class ProgramProper:
    BUTTON_PIN_1 = 17
    BUTTON_PIN_2 = 18

    def __init__(self):
        self.AUTO_MODE = True
        self.AUTO_COUNTER = time.time()
        self.AUTO_INTERVAL = 10

    def setup_gpio(self):
        """Set up GPIO pins and event detection."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.BUTTON_PIN_1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.BUTTON_PIN_2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def interaction(self, classes):
        freq = Counter(classes)

        try:
            if self.AUTO_MODE:
                if GPIO.input(self.BUTTON_PIN_1) == GPIO.LOW:  # Cycles through intervals 10, 20 and 30 seconds
                    if self.AUTO_INTERVAL == 10:
                        print("20-second interval")
                        self.AUTO_INTERVAL = 0
                        tts = gTTS("20-second interval")
                        tts.save("auto_20sec.mp3")
                        time.sleep(.5)
                        os.system("mpg321 auto_20sec.mp3")
                        self.AUTO_INTERVAL = 20
                    elif self.AUTO_INTERVAL == 20:
                        print("30-second interval")
                        self.AUTO_INTERVAL = 0
                        tts = gTTS("30-second interval")
                        tts.save("auto_30sec.mp3")
                        time.sleep(.5)
                        os.system("mpg321 auto_30sec.mp3")
                        self.AUTO_INTERVAL = 30
                    else:
                        self.AUTO_INTERVAL = 0
                        print("10-second interval")
                        tts = gTTS("10-second interval")
                        tts.save("auto_10sec.mp3")
                        time.sleep(.5)
                        os.system("mpg321 auto_10sec.mp3")
                        self.AUTO_INTERVAL = 10
                if GPIO.input(self.BUTTON_PIN_2) == GPIO.LOW:
                    self.AUTO_MODE = False
                    print("Manual Mode")
                    os.system("mpg321 manual_mode.mp3")
                if time.time() - self.AUTO_COUNTER >= self.AUTO_INTERVAL:
                    self.AUTO_COUNTER = time.time()
                    if bool(freq):
                        voice = ", and ".join([f"{count} {item}" for item, count in freq.items()])
                        tts = gTTS(text=voice)
                        tts.save("detect.mp3")
                        time.sleep(.5)
                        os.system("mpg321 detect.mp3")
                        print(freq)
            else:
                if GPIO.input(self.BUTTON_PIN_1) == GPIO.LOW:
                    self.AUTO_MODE = True
                    print("Auto Mode")
                    os.system("mpg321 auto_mode.mp3")
                if GPIO.input(self.BUTTON_PIN_2) == GPIO.LOW:
                    voice = ", and ".join([f"{count} {item}" for item, count in freq.items()])
                    tts = gTTS(text=voice)
                    tts.save("detect.mp3")
                    time.sleep(.5)
                    os.system("mpg321 detect.mp3")
                    print(freq)

        except AssertionError:
            pass

    def main(self):
        """Main function."""
        args = self.parse_arguments()
        self.setup_gpio()
        detector = ObjectDetector(args.model, args.margin, args.row_size, args.font_size, args.font_thickness,
                                  args.text_color)
        detector.run_inference(args.cameraId, args.frameWidth, args.frameHeight, args.numThreads, args.enableEdgeTPU)

    @staticmethod
    def parse_arguments():
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--model', help='Path of the object detection model.', default='efficientdet_lite0.tflite')
        parser.add_argument('--cameraId', help='Id of camera.', type=int, default=0)
        parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', type=int, default=640)
        parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', type=int, default=480)
        parser.add_argument('--numThreads', help='Number of CPU threads to run the model.', type=int, default=4)
        parser.add_argument('--enableEdgeTPU', help='Whether to run the model on EdgeTPU.', action='store_true',
                            default=False)
        parser.add_argument('--margin', help='Margin for bounding box visualization.', type=int, default=10)
        parser.add_argument('--rowSize', help='Size of row for text display.', type=int, default=10)
        parser.add_argument('--fontSize', help='Font size for text display.', type=int, default=1)
        parser.add_argument('--fontThickness', help='Font thickness for text display.', type=int, default=1)
        parser.add_argument('--textColor', help='Text color for text display.', type=tuple, default=(52, 29, 197))
        args = parser.parse_args()

        return args

if __name__ == '__main__':
    program = ProgramProper()
    program.main()