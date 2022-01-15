import cv2
import random

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_detector = cv2.CascadeClassifier("haarcascade_smile.xml")

webcam = cv2.VideoCapture(0)

while True:
    success, frame = webcam.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(grayscale)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (random.randint(0,255), random.randint(0,255), random.randint(0,255)), 4)
    
    cv2.imshow("Smile Detector!",frame)
    cv2.waitKey(1)
    
webcam.release()

