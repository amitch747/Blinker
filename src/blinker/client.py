import cv2
import mediapipe as mp
import pyautogui
pyautogui.FAILSAFE = False
import time
import socket
import threading
from pynput import keyboard

from ear import EAR_check, calc_thresh
from cursor import cursor, calibrate_cursor, move_mouse
from utils import draw_text_pil

from settings import CAMERA_INDEX, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE
from settings import L_COOLDOWN, R_COOLDOWN, DEFAULT_L_THRESH, DEFAULT_R_THRESH

LEFT_EYE_LANDMARKS = [362, 380, 373, 263, 387, 385]
LEFT_IRIS_LANDMARKS  = [474, 475, 476, 477]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

leftEyePos = []
rightEyePos = []

KEY_STATE = {'q': False, 't': False, 'c': False, 'm': False}
def on_press(key):
    try:
        if key.char in KEY_STATE:
            KEY_STATE[key.char] = True
    except AttributeError:
        pass


class LeftEye:
    def __init__(self):
        self.toggleFlag = False
        self.closedFlag = False
        self.lastClick = 0
        self.coolDown = L_COOLDOWN
        self.thresh = DEFAULT_L_THRESH

    def toggle(self):
        self.toggleFlag = not self.toggleFlag
        # print(f"Toggle: {self.toggleFlag}")
    

    def checkLeftEye(self, leftEyePos):
        if self.toggleFlag:
            if EAR_check(leftEyePos, self.thresh):
                if not self.closedFlag:
                    self.closedFlag = True
                    pyautogui.mouseDown()
                    print('left mouse down (tog)')
            else:
                if self.closedFlag:
                    self.closedFlag = False
                    pyautogui.mouseUp()
                    print('left mouse up (tog)')
                    self.closedFlag = False
            return
        
        ear = EAR_check(leftEyePos, self.thresh)
        if ear == True:
            if not self.closedFlag and (time.time() - self.lastClick) > self.coolDown:
                pyautogui.click()
                self.closedFlag = True
                self.lastClick = time.time()
                print('left click')
        elif ear == False:
            self.closedFlag = False
        else:
            # must have been none
            return 
class RightEye:     
    def __init__(self):
        self.closedFlag = False
        self.lastClick = 0 
        self.coolDown = R_COOLDOWN
        self.thresh = DEFAULT_R_THRESH

    def checkRightEye(self, rightEyePos):
        ear = EAR_check(rightEyePos, self.thresh)
        if ear == True:
            if not self.closedFlag and (time.time() - self.lastClick) > self.coolDown:
                pyautogui.rightClick()
                self.closedFlag = True
                self.lastClick = time.time()
                print('right click')
        elif ear == False:
            self.closedFlag = False   
        else:
            # must have been none
            return      
  

def calibrate_eye(leftEye, rightEye):
    # Reset EAR thresholds
    # 0.02 is placeholder will be configurable
    leftEye.thresh = calc_thresh(leftEyePos) + 0.02
    rightEye.thresh = calc_thresh(rightEyePos) + 0.02


def main():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('127.0.0.1', 747))    
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode = False,
        max_num_faces = 1,
        refine_landmarks = True, 
        min_detection_confidence = DETECTION_CONFIDENCE,
        min_tracking_confidence = TRACKING_CONFIDENCE
    )
    
    leftEye = LeftEye()             
    rightEye = RightEye()
    
    frameCount = 0  
    fps = 0
    startTime = time.time()
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    CURSOR_TOGGLE = False
    threading.Thread(target=move_mouse, args=(client,), daemon=True).start()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failure to read from camera")
            continue
        leftEyePos.clear()
        rightEyePos.clear()
        # Returns a NamedTuple object with a "multi_face_landmarks" field 
        # that contains the face landmarks on each detected face.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        
        fofo = None
        fofi = None
        fosi = None
        fosev = None
        
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in LEFT_EYE_LANDMARKS:
                    # Convert to coorinates of frame
                    x = int(landmark.x * frame.shape[1]) 
                    y = int(landmark.y * frame.shape[0]) 
                    leftEyePos.append((x,y))

                    if leftEye.closedFlag == True :
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), 10)  
                    else:
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  
    
                    
                elif idx in RIGHT_EYE_LANDMARKS:
                    x = int(landmark.x * frame.shape[1]) 
                    y = int(landmark.y * frame.shape[0]) 
                    rightEyePos.append((x,y))
                    if rightEye.closedFlag == True:
                        cv2.circle(frame, (x, y), 2, (50, 0, 255), 10)
                    else:
                        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                        
                elif idx in LEFT_IRIS_LANDMARKS:
                    if leftEye.toggleFlag == True:

                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
                        if idx == 474:
                            fofo = (x,y)
                        if idx == 475:
                            fofi = (x,y)
                        if idx == 476:
                            fosi = (x,y)
                        if idx == 477:
                            fosev = (x,y)
  
            if CURSOR_TOGGLE:
                cursor(frame, face_landmarks)
            rightEye.checkRightEye(rightEyePos)
            leftEye.checkLeftEye(leftEyePos)
            
        if fofo is not None and fosi is not None:
            cv2.line(frame, fofo, fosi, (0, 255, 255), 1) 
        if fofi is not None and fosev is not None:
            cv2.line(frame, fofi, fosev, (0, 255, 255), 1) 
 
        frameCount += 1
        if (time.time() - startTime) >= 1.0:
            fps = frameCount
            startTime = time.time()
            frameCount = 0
                
        draw_text_pil(frame, f" 'm' toggle cursor |'t' toggle left | 'c' calibrate | 'q' quit", (10, 425), (1, 174, 255))
        draw_text_pil(frame, f"FPS: {fps}", (500, 20), (1, 174, 255))


        cv2.namedWindow('Blinker', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Blinker', cv2.WND_PROP_TOPMOST, 1)

        cv2.imshow('Blinker', frame)
        cv2.waitKey(1)
        
        if KEY_STATE['q']:
            break
        if KEY_STATE['t']:
            leftEye.toggle()
            KEY_STATE['t'] = False
        if KEY_STATE['c']:
            calibrate_eye(leftEye, rightEye)
            calibrate_cursor()
            KEY_STATE['c'] = False
        if KEY_STATE['m']:
            CURSOR_TOGGLE = not CURSOR_TOGGLE
            KEY_STATE['m'] = False
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()