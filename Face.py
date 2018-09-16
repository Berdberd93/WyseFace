import face_recognition
import picamera
import numpy as np
import cv2
import time

camera = picamera.PiCamera()
camera.resolution = (320, 240)
map = np.empty((240, 320, 3), dtype=np.uint8)

# Initialize some variables
face_locations = []

for foo in camera.capture_continuous(map, format="rgb", use_video_port=True):
    print("Capturing image.")
    start = time.time()
    image = map
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(image)
    print("Found {} faces in image.".format(len(face_locations)))
    
    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(1)
    print(1/(time.time()-start))
