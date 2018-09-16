import dlib # dlib for accurate face detection
import cv2 # opencv
import picamera
import numpy as np
import time
from threading import Thread
 
class PiStream:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.camera = picamera.PiCamera()
        self.camera.resolution = (320, 240)
        self.camera.framerate = 64
        self.rawCapture = np.empty((240, 320, 3), dtype=np.uint8)
        self.output = None
        self.update_time = 0.27/3
        self.stream = self.camera.capture_continuous(self.rawCapture, format="rgb", use_video_port=True)
        
        self.stopwatch = 0

        #Thread(target=self.capture, args=()).start()           
    
    def display(self):
        if self.output is not None:
            # show the frame
            cv2.imshow("Face Detection", self.output)
            key = cv2.waitKey(1)
            #print("Display liao")
            
        
    def run(self):
        for foo in self.camera.capture_continuous(self.rawCapture, format="rgb", use_video_port=True):
            start = time.time()
            #self.image = self.rawCapture
            #print("Captured")
            self.control()
            self.display()
            print("Time: {} \t".format((time.time()-start)))
            
          
    def control(self):
        """
        Future improvement: (self.update_time/3)
        Get a list of update_time and get the median
        """
        if (time.time()-self.stopwatch) > (self.update_time*3):
            print("Update: {}".format(self.update_time))
            self.stopwatch = time.time()
            Thread(target=self.update, args=()).start()
       
    def update(self):
        #print("Update")
        update_start = time.time()
        gray = cv2.cvtColor(self.rawCapture, cv2.COLOR_RGB2GRAY)
     
        # Make copies of the frame for transparency processing
        overlay = self.rawCapture.copy()
        output = self.rawCapture.copy()
     
        # set transparency value
        alpha  = 0.5
     
        # detect faces in the gray scale frame
        #face = time.time()
        face_rects = self.detector(gray, 0)
        #print("Face: {}".format(time.time()-face))
     
        # loop over the face detections
        for i, d in enumerate(face_rects):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
     
            # draw a fancy border around the faces
            self.draw_border(overlay, (x1, y1), (x2, y2), (162, 255, 0), 2, 10, 10)
     
        # make semi-transparent bounding box
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        self.output = output
        self.update_time = time.time() - update_start
        #print("Update time: {}".format(self.update_time))
        
     
    # Fancy box drawing function by Dan Masek
    @staticmethod
    def draw_border(img, pt1, pt2, color, thickness, r, d):
        x1, y1 = pt1
        x2, y2 = pt2
     
        # Top left drawing
        cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
     
        # Top right drawing
        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
     
        # Bottom left drawing
        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
     
        # Bottom right drawing
        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
            
     
if __name__ == '__main__' :
    try:
        stream = PiStream()
        stream.run()
    finally:
        print("Device is forced to reset.")
        stream.camera.close()
        time.sleep(1)
        cv2.destroyAllWindows()
        #stream = self.stream
        #stream.stop()             

