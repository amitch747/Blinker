import cv2
import mediapipe as mp
import pyautogui
pyautogui.FAILSAFE = False
import time
import math
import socket
import threading
from pynput import keyboard
import numpy as np

from ear import calc_EAR

# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client.connect(('127.0.0.1', 54000))


WIDTH, HEIGHT = pyautogui.size()
SCREEN_CENTER_X = WIDTH // 2
SCREEN_CENTER_Y = HEIGHT // 2
MOUSE_LOCATION = [SCREEN_CENTER_X, SCREEN_CENTER_Y]
DEADZONE = 10

MOUSE_MUTEX = threading.Lock()

FACE_LANDMARKS = {1:"NOSE",10:"TOP",152:"BOTTOM",234:"LEFT",454:"RIGHT"}
LEFT_EYE_LANDMARKS = [362, 380, 373, 263, 387, 385]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

neutral_x = None
neutral_y = None


KEY_STATE = {'q': False, 't': False, 'c': False}
def on_press(key):
    try:
        if key.char in KEY_STATE:
            KEY_STATE[key.char] = True
    except AttributeError:
        pass

def mouse_mover():
    while True:
        with MOUSE_MUTEX:
            x, y = MOUSE_MUTEX
        pyautogui.moveTo(x, y)
        time.sleep(0.01)  



class LeftEye:
    def __init__(self):
        self.toggleFlag = False
        self.closedFlag = False
        self.lastClick = 0
        self.coolDown = 0.7

    def toggle(self):
        self.toggleFlag = not self.toggleFlag
        # print(f"Toggle: {self.toggleFlag}")

    def checkLeftEye(self, leftEyePos):
        if len(leftEyePos) != 6:
            return  

        if self.toggleFlag:
            if calc_EAR(leftEyePos, 0.8):
                pyautogui.click()
                print('left click (tog)')
            return

        if calc_EAR(leftEyePos, 0.75):
            if not self.closedFlag and (time.time() - self.lastClick) > self.coolDown:
                pyautogui.click()
                self.closedFlag = True
                self.lastClick = time.time()
                print('left click')
        else:
            self.closedFlag = False
class RightEye:     
    def __init__(self):
        self.closedFlag = False
        self.lastClick = 0 
        self.coolDown = 2

    def checkRightEye(self, rightEyePos):
        if len(rightEyePos) != 6:
            return  

        if calc_EAR(rightEyePos, 0.8):
            if not self.closedFlag and (time.time() - self.lastClick) > self.coolDown:
                pyautogui.rightClick()
                self.closedFlag = True
                self.lastClick = time.time()
                print('right click')
        else:
            self.closedFlag = False      
  



  
def cursor(frame,face_landmarks):
    frame_height, frame_width = frame.shape[:2]
    raw_face_data = {}
    
    for idx, landmark in enumerate(face_landmarks.landmark):
        if idx in FACE_LANDMARKS:
            # Convert to coorinates of frame
            location = FACE_LANDMARKS[idx]
            x = int(landmark.x * frame_width) 
            y = int(landmark.y * frame_height) 
            z = int(landmark.z * frame_width) # z is in image-width units for some reason
            raw_face_data[location] = np.array([x,y,z])

            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)

    # seperate face data
    nose = raw_face_data["NOSE"]
    top = raw_face_data["TOP"]
    bottom = raw_face_data["BOTTOM"]
    left = raw_face_data["LEFT"]
    right = raw_face_data["RIGHT"]
    
    right_vec = (right-left)
    right_vec_norm = np.linalg.norm(right_vec)
    right_unit_vec = right_vec/right_vec_norm
    
    up_vec = (top-bottom)
    up_vec_norm = np.linalg.norm(up_vec)
    up_unit_vec = up_vec/up_vec_norm
    
    out_vec = np.cross(right_unit_vec, up_unit_vec) # cross prod gives orthogonal vector out of face
    out_vec_norm = np.linalg.norm(out_vec)
    out_unit_vec = out_vec/out_vec_norm
    
    face_center = (left + right + top + bottom + nose) / 5
    
    ray_end = face_center - out_unit_vec * 300

    cv2.line(frame, tuple(face_center [:2].astype(int)), tuple(ray_end[:2].astype(int)), (15,255,0), 3)

    # if (neutral_x is None or neutral_y is None):
    #     neutral_x = x
    #     neutral_y = y
    #     print(f"Recalibrated neutral at x={neutral_x:.2f}, y={neutral_y:.2f}")
    # else:
    #     dx = x - neutral_x
    #     dy = y - neutral_y


    #     cv2.circle(frame, (int(neutral_x), int(neutral_y)), int(DEADZONE), (0, 0, 255), 1)
    #     cv2.circle(frame, (int(neutral_x), int(neutral_y)), int(DEADZONE*3), (0, 0, 255), 1)
    #     cv2.circle(frame, (int(neutral_x), int(neutral_y)), int(DEADZONE*6), (0, 0, 255), 1)

    #     vec = math.sqrt(abs(dx)**2 + abs(dy)**2 ) 

    #     if (vec < DEADZONE*3):
    #         SENSITIVITY = 3.0
    #     elif (DEADZONE*3 < vec < DEADZONE*6) :
    #         SENSITIVITY = 4.0
    #     elif (vec > DEADZONE*6):
    #         SENSITIVITY = 5.0
        
    #     cv2.putText(frame, "Sens:"+str(SENSITIVITY), (10, 120),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #     cv2.putText(frame, "Vec:"+str(vec), (10, 140),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #     # Apply deadzone
    #     if abs(dx) < DEADZONE:
    #         dx = 0
    #     if abs(dy) < DEADZONE:
    #         dy = 0

    #     # Apply sensitivity
    #     move_x = -int(dx * SENSITIVITY)
    #     move_y = -int(-dy * SENSITIVITY)  # Invert Y so down head = down mouse
        
    #     cv2.circle(frame, (int(neutral_x), int(neutral_y)), 4, (0, 0, 255), -1)
        
    #     cv2.line(frame, (int(neutral_x), int(neutral_y)), (int(x), int(y)), (0, 255, 0), 2)

    #     mouse_x, mouse_y = pyautogui.position()

    #     # message = f"{mouse_x} {mouse_y}" 
    #     # client.sendall(message.encode())
    
    #     # message = f"{move_x} {move_y}"
    #     # client.sendall(message.encode())
        
    #     with mouse_lock:
    #         mouse_target[0] = max(0, min(screen_width, mouse_x + move_x))
    #         mouse_target[1] = max(0, min(screen_height, mouse_y + move_y))
        
        # if (EDGE_MARGIN < mouse_x < screen_width - EDGE_MARGIN and
        #     EDGE_MARGIN < mouse_y < screen_height - EDGE_MARGIN):
        #     if move_x != 0 or move_y != 0:
        #         pyautogui.moveRel(move_x, move_y)

def main():
    # Init mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    leftEye = LeftEye()             
    rightEye = RightEye()
    
    frameCount = 0  
    fps = 0
    startTime = time.time()
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    threading.Thread(target=mouse_mover, daemon=True).start()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failure to read from camera")
            continue

        # Returns a NamedTuple object with a "multi_face_landmarks" field 
        # that contains the face landmarks on each detected face.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        leftEyePos = []
        rightEyePos = []
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in LEFT_EYE_LANDMARKS:
                    # Convert to coorinates of frame
                    x = int(landmark.x * frame.shape[1]) 
                    y = int(landmark.y * frame.shape[0]) 
                    leftEyePos.append([x,y])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  
                elif idx in RIGHT_EYE_LANDMARKS:
                    x = int(landmark.x * frame.shape[1]) 
                    y = int(landmark.y * frame.shape[0]) 
                    rightEyePos.append([x,y])
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                    
            cursor(frame, face_landmarks)
            rightEye.checkRightEye(rightEyePos)
            leftEye.checkLeftEye(leftEyePos)
 
        
        frameCount += 1
        if (time.time() - startTime) >= 1.0:
            fps = frameCount
            startTime = time.time()
            frameCount = 0
                


        cv2.putText(frame, " 't' - toggle left | 'q' - quit", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Left Toggle: {leftEye.toggleFlag}", (10,60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Left: {leftEye.closedFlag}, Right: {rightEye.closedFlag}", (10,300), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

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
            global neutral_x, neutral_y            
            neutral_x = None
            neutral_y = None
            print("Recalibrated neutral point.")
            KEY_STATE['c'] = False

            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()