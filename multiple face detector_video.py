import cv2
trained_face_data=cv2.CascadeClassifier('D:/PYTHON PROGRAMS/COMPUTER VISION/Computer-Vision/haarcascade_frontalface_default.xml')
webcam=cv2.VideoCapture(0)

while True:
    successful_frame_read, frame=webcam.read()
    grayscaled_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.imshow('Face Detector', frame)
    key=cv2.waitKey(1)
    if key == 81 or key == 113:
        break

webcam.release()

print('\n Code Completed')

