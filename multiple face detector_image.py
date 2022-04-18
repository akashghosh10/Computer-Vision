import cv2
trained_face_data=cv2.CascadeClassifier('D:/PYTHON PROGRAMS/OPENCV/Face detector/haarcascade_frontalface_default.xml')
img=cv2.imread('D:/PYTHON PROGRAMS/OPENCV/Face detector/face4.png')
grayscaled_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)
print(face_coordinates)
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
cv2.imshow('Face Detector', img)
cv2.waitKey()
print('\n Code Completed')