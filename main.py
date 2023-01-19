import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel,
                             QLineEdit, QVBoxLayout, QFrame, QInputDialog)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5 import QtCore
from datetime import datetime
import pathlib
import numpy as np
import cv2 as cv
import pyttsx3
import shutil


class Main_Window(QWidget):
    """A class that supports the program's main window"""
    def __init__(self):
        super().__init__()

        self.resize(600, 300)

        self.setWindowTitle("Face recognition")

        """Adding gui elements"""
        self.Recognition_Button = QPushButton("Recognize the face")
        self.Recognition_Button.clicked.connect(self.open_recognition)

        self.Add_Face = QPushButton("Add a face to the base")
        self.Add_Face.clicked.connect(self.add_new_face)

        self.Window_Face = QLabel()
        self.Window_Face.setFrameShape(QFrame.WinPanel)
        self.Window_Face.setFrameShadow(QFrame.Plain)

        self.Window_Face.setLineWidth(6)
        self.Window_Face.setMidLineWidth(6)
        self.Window_Face.setText("")
        self.Window_Face.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)

        self.Recognition_Status = QLineEdit()
        self.Recognition_Status.setReadOnly(True)

        self.Image = QPixmap()

        """Creating a Box layout"""
        layout = QVBoxLayout()

        layout.addWidget(self.Recognition_Button)
        layout.addWidget(self.Add_Face)
        layout.addWidget(self.Window_Face)
        layout.addWidget(self.Recognition_Status)

        self.setLayout(layout)

        """Adding a font and size"""
        self.Recognition_Button.setFont(QFont('Times New Roman', 14))
        self.Add_Face.setFont(QFont('Times New Roman', 14))
        self.Recognition_Status.setFont(QFont('Times New Roman', 14))

        """Loading the project folder and adding variables"""
        self.project_path = pathlib.Path().resolve()
        """Specifies the path to the Haar cascade classifiers of the OpenCV library.
        The classifier is installed by default with the library, for the project it is located in the project folder"""
        self.face_detector = cv.CascadeClassifier(f"{self.project_path}\\haarcascade_frontalface_default.xml")
        self.name = ''
        self.ID = ''

        """Specifies settings for audio instructions, you can change the voice to something other than default"""
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 145)
        self.engine.setProperty('volume', 1.0)

        if os.path.exists("Data.json"):
            try:
                with open("Data.json") as json_file:
                    self.data = json.load(json_file)
            except:
                self.Recognition_Status.setText("Failed to open a database of people")

        else:
            self.data = {}

        """Creating folders if there are none"""
        self.exist_dict()

    def open_recognition(self):
        self.take_photos(5)
        self.face_predict()

    def face_predict(self):
        """Determines algorithm settings for face recognition and uploads training data."""
        recognizer = cv.face.LBPHFaceRecognizer_create()

        """Specifies test data settings."""
        recognizer_path = f"{self.project_path}\\LBPHFaceRecognizer"
        recognizer_paths = [os.path.join(recognizer_path, f) for f in os.listdir(recognizer_path)]
        image_path = f"{self.project_path}\\PhotosToRecognize"
        image_paths = [os.path.join(image_path, f) for f in os.listdir(image_path)]

        """It creates a loop through images and predicts identity."""
        if len(recognizer_paths) != 0:
            for image in image_paths:
                for LBPHF in recognizer_paths:
                    recognizer.read(LBPHF)
                    predict_image = cv.imread(image, cv.IMREAD_GRAYSCALE)
                    faces = self.face_detector.detectMultiScale(predict_image,
                                                                scaleFactor=1.05, minNeighbors=10)
                    for (x, y, w, h) in faces:
                        face = cv.resize(predict_image[y:y + h, x:x + w], (100, 100))
                        predicted_id, dist = recognizer.predict(face)
                        if predicted_id == 1 and dist <= 95:
                            name = self.data[predicted_id]
                            print("Person in the picture {}: {} (distance = {})" .format(image, name, round(dist, 1)))
                            print(f"Access allowed (user: {name}, data:" 
                                  f" {datetime.now()}).", file=open('lab_access_log.txt', 'a'))
                        else:
                            name = 'not recognized'
                            self.Recognition_Status.setText(f"The person in the picture {image}: {name}")

                        cv.rectangle(predict_image, (x, y), (x + w, y + h), 255, 2)
                        cv.putText(predict_image, name, (x + 1, y + h - 5),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                        cv.imshow('ID', predict_image)
                        cv.waitKey(2000)
                        cv.destroyAllWindows()
        else:
            self.Recognition_Status.setText("No people in the database, recognition not started")

        if len(image_paths) != 0:
            for image in image_paths:
                os.remove(image)

    def take_photos(self, number_of_photos=5, name='', ID=''):
        """Prepares the camera. If you have more than one camera you can change the argument 0 to another. """
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            self.Recognition_Status("Unable to start video device.")
        cap.set(3, 640)
        cap.set(4, 480)

        """Provides instructions for setting up faces for photos."""
        messeage = "Rremove your glasses and other items covering your face after which" \
                   " look directly into the camera. " \
                   "Assume different facial expressions. In a few seconds we will start receiving images"
        self.engine.say(messeage)
        self.engine.runAndWait()

        """Selects a folder to save photos for recognition """
        os.chdir(f"{self.project_path}\\PhotosToRecognize")
        frame_count = 0

        while frame_count < number_of_photos:
            """It acquires images frame by frame thirty times."""
            _, frame = cap.read()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            """You can change the minNeighbors parameter to speed up the time, 
            but too small a value may cause false results """
            face_rects = self.face_detector.detectMultiScale(gray, scaleFactor=1.2,
                                                             minNeighbors=10)
            for (x, y, w, h) in face_rects:
                frame_count += 1
                if (name and ID) != '':
                    photo_name = f"{name}_{ID}.{str(frame_count)}.jpg"
                else:
                    photo_name = f"Photo.{str(frame_count)}.jpg"
                cv.imwrite(photo_name, gray[y:y + h, x:x + w])
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                image = QPixmap(photo_name)
                self.Window_Face.setPixmap(image)
                cv.waitKey(200)

        self.Recognition_Status.setText("Finished taking pictures")
        cap.release()
        os.chdir(self.project_path)

    def add_new_face(self):

        name, okPressed = QInputDialog.getText(self, "Get name", "Your name:", QLineEdit.Normal, "")
        if okPressed and name != '':
            self.name = name

        ID, okPressed = QInputDialog.getText(self, "Get ID", "Your ID:", QLineEdit.Normal, "")
        if okPressed and ID != '':
            self.ID = ID

        if (self.ID and self.name) != '':
            self.take_photos(30, self.name, self.ID)
            self.trainer()
            self.create_dict()
            self.move_photos()
            self.data[self.name] = self.ID
            json_data = json.dumps(self.data)
            with open("Data.json", "w") as outfile:
                outfile.write(json_data)
            self.ID = ''
            self.name = ''

    def trainer(self):

        train_path = f"{self.project_path}\\PhotosToRecognize"

        image_paths = [os.path.join(train_path, f) for f in os.listdir(train_path)]
        images, labels = [], []

        """Finds rectangles with faces and assigns them labels in the form of numbers."""
        for image in image_paths:
            train_image = cv.imread(image, cv.IMREAD_GRAYSCALE)
            label = int(os.path.split(image)[-1].split('.')[1])
            faces = self.face_detector.detectMultiScale(train_image)
            for (x, y, w, h) in faces:
                images.append(train_image[y:y + h, x:x + w])
                labels.append(label)
                cv.imshow("Training image", train_image[y:y + h, x:x + w])
                cv.waitKey(50)

        cv.destroyAllWindows()

        """Conducts training of the algorithm"""
        recognizer = cv.face.LBPHFaceRecognizer_create()
        recognizer.train(images, np.array(labels))
        recognizer.write(f"{self.project_path}\\LBPHFaceRecognizer\\{self.name}_{self.ID}_lbph.yml")
        self.Recognition_Status.setText("Training completed")

    def move_photos(self):
        source = f"{self.project_path}\\PhotosToRecognize"
        destination = f"{self.project_path}\\PhotoDatabase\\{self.name}_{self.ID}"

        """gather all files"""
        images = os.listdir(source)

        """iterate on all files to move them to destination folder"""
        for image in images:
            src_path = os.path.join(source, image)
            dst_path = os.path.join(destination, image)
            shutil.move(src_path, dst_path)

    def create_dict(self):
        if os.path.isdir(f"{self.project_path}\\PhotoDatabase\\{self.name}_{self.ID}"):
            pass
        else:
            os.makedirs(f"{self.project_path}\\PhotoDatabase\\{self.name}_{self.ID}")

    def exist_dict(self):
        """Create folders if they don't exist"""

        if os.path.isdir(f"{self.project_path}\\LBPHFaceRecognizer"):
            pass
        else:
            os.makedirs(f"{self.project_path}\\LBPHFaceRecognizer")

        if os.path.isdir(f"{self.project_path}\\PhotosToRecognize"):
            pass
        else:
            os.makedirs(f"{self.project_path}\\PhotosToRecognize")

        if os.path.isdir(f"{self.project_path}\\PhotoDatabase"):
            pass
        else:
            os.makedirs(f"{self.project_path}\\PhotoDatabase")


if __name__ == '__main__':

    app = QApplication(sys.argv)
    okno = Main_Window()
    okno.show()
    sys.exit(app.exec_())
