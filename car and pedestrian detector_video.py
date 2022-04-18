import cv2
trained_car_data=cv2.CascadeClassifier('D:/PYTHON PROGRAMS/COMPUTER VISION/Computer-Vision/cars.xml')
trained_pedestrian_data=cv2.CascadeClassifier('D:/PYTHON PROGRAMS/COMPUTER VISION/Computer-Vision/haarcascade_fullbody.xml')
video=cv2.VideoCapture('cars_and_pedestrians_video_1.mp4')
while True:
    successful_frame_read, frame=video.read()
    grayscaled_video=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    car_coordinates=trained_car_data.detectMultiScale(grayscaled_video)
    pedestrian_coordinates=trained_pedestrian_data.detectMultiScale(grayscaled_video)
    for (x, y, w, h) in car_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in pedestrian_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Car Detector', frame)
    key=cv2.waitKey(1)
    if key == 81 or key == 113:
        break

video.release()

print('\n Code Completed')