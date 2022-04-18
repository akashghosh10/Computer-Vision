import cv2
trained_face_data=cv2.CascadeClassifier('D:/PYTHON PROGRAMS/COMPUTER VISION/Computer-Vision/haarcascade_frontalface_default.xml')
trained_smile_data=cv2.CascadeClassifier('D:/PYTHON PROGRAMS/COMPUTER VISION/Computer-Vision/haarcascade_smile.xml')
webcam=cv2.VideoCapture(0)

while True:
    successful_frame_read, frame=webcam.read()
    grayscaled_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face=frame[y:y+h, x:x+h]
        grayscaled_face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        smile=trained_smile_data.detectMultiScale(grayscaled_face, scaleFactor=1.7, minNeighbors=15)
        for (x_, y_, w_, h_) in smile:
            cv2.rectangle(face, (x_, y_), (x_+w_, y_+h_), (255, 0, 0), 1)
        if len(smile)>0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
    cv2.imshow('Smile Detector', frame)
    key=cv2.waitKey(1)
    if key == 81 or key == 113:
        break

webcam.release()

print('\n Code Completed')

