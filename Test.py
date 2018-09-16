import dlib # dlib for accurate face detection
import cv2 # opencv
from picamera import PiCamera
import numpy as np
import time
from threading import Thread

import io
import socket
import struct
 
class PiStream:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.camera = PiCamera()
        self.camera.resolution = (320, 240)
        self.camera.framerate = 32
        #self.camera.brightness = 80
        #self.camera.contrast = 80
        self.rawCapture = np.empty((240, 320, 3), dtype=np.uint8)
        self.output = None
        self.update_time = 0.27/3
       
        self.bb_x1 = 80
        self.bb_y1 = 30
        self.bb_x2 = 240
        self.bb_y2 = 210
        self.bbx_tolerance = 25
        self.bby_tolerance = 20
        self.iou_threshold = 0.5
       
        self.location = list()
        self.decision = False
        self.update_stopwatch = 0
        self.cli_busy_stopwatch = 0
        
        # Connect a client socket to my_server:8000 (change my_server to the
        # hostname of your server)
        client_image_socket = socket.socket()
        client_image_socket.connect(('192.168.43.149', 8000))

        #client_face_socket = socket.socket()
        #client_image_socket.connect(('192.168.43.149', 6000))

        # Make a file-like object out of the connection
        self.image_connection = client_image_socket.makefile('wb')
        #self.face_connection = client_face_socket.makefile('rb')
        
        self.client_busy = False
            
        
    def run(self):
        for foo in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            start = time.time()
            #self.rawCapture = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.control()
            if self.decision:
                self.send_server()
            #self.read_server()
            self.display()
            #print("Time: {} \t".format((time.time()-start)))
            
        # Write a length of zero to the stream to signal we're done
        connection.write(struct.pack('<L', 0))
         
    def read_server(self):
        #data_stream = io.BytesIO()
        
        #data_len = struct.unpack('<L', self.face_connection.read(struct.calcsize('<L')))[0]
        #if data_len:
        #    return
        #data_stream.write(self.face_connection.read(1024))
        #data_stream.seek(0)
        #data = data_stream.read()
        #data_stream.seek(0)
        #data_stream.truncate()
        #print(str(data))
        message = self.client_image_socket.recv(2048)
        if not message:
            return
        print(str(message))
                                 

    def send_server(self, image):
        #print("IoU: {}".format(iou))
        image_stream = self.image_to_stream(image)
        
        # Send stream to server
        self.image_connection.write(struct.pack('<L', image_stream.tell()))
        self.image_connection.flush()
        
        # Rewind the stream and send the image data over the wire
        image_stream.seek(0)
        self.image_connection.write(image_stream.read())

        # Reset the stream for the next capture
        image_stream.seek(0)
        image_stream.truncate()
        
    def image_to_stream(self, image):        
        bytes = np.asarray(bytearray(cv2.imencode(".jpeg", image)[1].tobytes()), dtype = np.uint8)        
        image_stream = io.BytesIO()        
        image_stream.write(bytes)
        return image_stream
        
          
    def control(self):
        """
        Future improvement: (self.update_time/3)
        Get a list of update_time and get the median
        """        
        if (time.time()-self.update_stopwatch) > (self.update_time):
            self.update_stopwatch = time.time()
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
        face_rects = self.detector(gray, 0)
         
        # loop over the face detections
        x1, y1, x2, y2 = 0, 0, 0, 0
        for i, d in enumerate(face_rects):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
     
            # draw a fancy border around the faces
            self.draw_border(overlay, (x1, y1), (x2, y2), (162, 255, 0), 2, 10, 10)
                        
            # Calculate IoU
            iou = self.calculate_IoU(x1, y1, x2, y2)            
            
            if iou is None:
                iou = 0
                
            if time.time() - self.cli_busy_stopwatch > 0.1:
                self.client_busy = False
            
            #print(str(time.time() - self.cli_busy_stopwatch))
            
            if iou > self.iou_threshold and self.client_busy is False:
                self.cli_busy_stopwatch = time.time()
                self.client_busy = True                

                #send_img = output[self.bb_y1: self.bb_y2, self.bb_x1: self.bb_x2]
                send_img = output[y1: y2, x1: x2]
    
                # Put send image to server
                self.send_server(send_img)
        
        self.draw_border(overlay, (self.bb_x1, self.bb_y1), (self.bb_x2, self.bb_y2), (0, 0, 255), 2, 10, 10)                   
        
        # make semi-transparent bounding box
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        self.output = output
        
        # Berd (Just for testing)
        #self.send_server(output)
        
        self.update_time = time.time() - update_start
        #print("Update time: {}".format(self.update_time))
        
    def check_intersection(self, x1, y1, x2, y2):
        status = False
        
        if ((x1 > self.bb_x1 - self.bbx_tolerance) and (x1 < self.bb_x2 + self.bbx_tolerance)) and ((x2 > self.bb_x1 - self.bbx_tolerance) and (x2 < self.bb_x2 + self.bbx_tolerance)) and ((y1 > self.bb_y1 - self.bby_tolerance) and (y1 < self.bb_y2 + self.bby_tolerance)) and ((y2 > self.bb_y1 - self.bby_tolerance) and (y2 < self.bb_y2 + self.bby_tolerance)):
            status = True        
        # print (str(status))
        return status            
        
    def calculate_IoU(self, x1, y1, x2, y2):
        # Check intersection betwwen pre-defined BB and generated BB
        if self.check_intersection(x1, y1, x2, y2):
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(x1, self.bb_x1)
            yA = max(y1, self.bb_y1)
            xB = min(x2, self.bb_x2)
            yB = min(y2, self.bb_y2)

            # compute the area of intersection rectangle
            interArea = (xB - xA) * (yB - yA)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (x2 - x1) * (y2 - y1)
            boxBArea = (self.bb_x2 - self.bb_x1) * (self.bb_y2 - self.bb_y1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = interArea / float(boxAArea + boxBArea - interArea)

            # return the intersection over union value
            return round(iou, 3)
    
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
        
    def display(self):
        if self.output is not None:
            # show the frame
            cv2.imshow("Face Detection", self.output)
            key = cv2.waitKey(1)
            #print("Display liao")
            
     
if __name__ == '__main__' :
    try:
        stream = PiStream()
        stream.run()
    finally:        
        print("Device is forced to reset.")
        stream.camera.close()
        time.sleep(2)
        cv2.destroyAllWindows()
        
        self.image_connection.close()
        #self.face_connection.close()
        self.client_image_socket.close()
        #self.client_face_socket.close()


