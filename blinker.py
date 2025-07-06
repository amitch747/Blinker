import cv2
import mediapipe as mp
import pyautogui
pyautogui.FAILSAFE = False
import time
import math
import socket
import threading
import numpy as np

from pynput import keyboard
from collections import deque

from ear import EAR_check, calc_thresh

from PIL import Image, ImageDraw, ImageFont
font = ImageFont.truetype("cs_regular.ttf", 24)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1', 54000))


WIDTH, HEIGHT = pyautogui.size()
SCREEN_CENTER_X = WIDTH // 2
SCREEN_CENTER_Y = HEIGHT // 2
MOUSE_LOCATION = [SCREEN_CENTER_X, SCREEN_CENTER_Y]
DEADZONE = 10

MOUSE_MUTEX = threading.Lock()

FACE_LANDMARKS = {1:"NOSE",10:"TOP",152:"BOTTOM",234:"LEFT",454:"RIGHT"}
LEFT_EYE_LANDMARKS = [362, 380, 373, 263, 387, 385]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

        
leftEyePos = []
rightEyePos = []

deque_len = 9
face_center_queue = deque(maxlen=deque_len)
out_vec_queue = deque(maxlen=deque_len)

calibration_yaw = 0
calibration_pitch = 0
raw_yaw = 180
raw_pitch = 180

KEY_STATE = {'q': False, 't': False, 'c': False}
def on_press(key):
    try:
        if key.char in KEY_STATE:
            KEY_STATE[key.char] = True
    except AttributeError:
        pass

def move_mouse():
    sensitivity = 0.05  # Adjust as needed
    while True:
        with MOUSE_MUTEX:
            x, y = MOUSE_LOCATION
            dx = int((x - SCREEN_CENTER_X) * sensitivity)
            dy = int((y - SCREEN_CENTER_Y) * sensitivity)
            # Reset the position to center after calculating delta
            MOUSE_LOCATION[0] = SCREEN_CENTER_X
            MOUSE_LOCATION[1] = SCREEN_CENTER_Y

        client.sendall(f"{dx} {dy}\n".encode())




class LeftEye:
    def __init__(self):
        self.toggleFlag = False
        self.closedFlag = False
        self.lastClick = 0
        self.coolDown = 0.7
        self.thresh = 0.8

    def toggle(self):
        self.toggleFlag = not self.toggleFlag
        # print(f"Toggle: {self.toggleFlag}")
    

    def checkLeftEye(self, leftEyePos):
        if len(leftEyePos) != 6:
            return  

        if self.toggleFlag:
            if EAR_check(leftEyePos, self.thresh):
                pyautogui.click()
                self.closedFlag = True
                print('left click (tog)')
            return

        if EAR_check(leftEyePos, self.thresh):
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
        self.coolDown = 1
        self.thresh = 0.8

    def checkRightEye(self, rightEyePos):
        if len(rightEyePos) != 6:
            return  

        if EAR_check(rightEyePos, self.thresh):
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
    out_unit_vec = -out_unit_vec # Out of face instead out of back of head
    
    
    face_center = (left + right + top + bottom + nose) / 5
    

    face_center_queue.append(face_center)
    out_vec_queue.append(out_unit_vec)

    avg_center = np.mean(face_center_queue, axis=0)
    avg_out = np.mean(out_vec_queue, axis=0)
    avg_out_norm = np.linalg.norm(avg_out)
    avg_out_unit = avg_out/avg_out_norm
    
    cam_out = np.array([0,0,-1])
    



    xz_out = np.array([avg_out_unit[0], 0, avg_out_unit[2]])
    xz_out /= np.linalg.norm(xz_out)
    yaw = np.degrees(math.acos(np.clip(np.dot(cam_out, xz_out), -1.0, 1.0)))
    if avg_out_unit[0] < 0:
        yaw = -yaw
    
    yz_out = np.array([0, avg_out_unit[1], avg_out_unit[2]])
    yz_out /= np.linalg.norm(yz_out)
    pitch = np.degrees(math.acos(np.clip(np.dot(cam_out, yz_out), -1.0, 1.0)))
    if avg_out_unit[1] > 0:
        pitch = -pitch 
        
        
    #print("yaw: {:.1f} pitch: {:.1f}".format(yaw, pitch))    # if (neutral_x is None or neutral_y is None):

    if yaw < 0:
        yaw = abs(yaw) # Negatives become positve
    elif yaw < 180:
        yaw = 360 - yaw # Positives are flipped
    # Thus we have full range
    if pitch < 0:
        pitch = 360 + pitch

    #print("yaw: {:.1f} pitch: {:.1f}".format(yaw, pitch))    # if (neutral_x is None or neutral_y is None):
    global raw_yaw, raw_pitch
    raw_yaw = yaw
    raw_pitch = pitch
    
    # Angles needed to reach screen bounds
    yaw_bound = 20
    pitch_bound = 10
    # For example, left of screen pixels will correspond to 180-yawDeg    

    yaw += calibration_yaw
    pitch += calibration_pitch

    # 180 - (160) / 40 = 0.5
    # 0.5 * WIDTH = screen center
    x_target = int(((yaw - (180 - yaw_bound)) / (2 * yaw_bound)) * WIDTH)
    # 180 + 10 - 90 / 20 = 5
    # 5 * HEIGHT = well past screen bound
    y_target = int(((180 + pitch_bound - pitch) / (2 * pitch_bound)) * HEIGHT)
    
    # Need to clamp bounds
    if y_target > HEIGHT - 10:
        y_target = HEIGHT - 10
    elif y_target < 10:
        y_target = 10
    if x_target >  WIDTH-10:
        x_target = WIDTH - 10
    elif x_target < 10:
        x_target = 10
        
    #print(f"Screen position: x={x_target}, y={y_target}")

    # Fill array
    with MOUSE_MUTEX:
        MOUSE_LOCATION[0] = x_target
        MOUSE_LOCATION[1] = y_target
        
    # 100 is arbitrary, should be configurable
    ray_end = avg_center - avg_out_unit * 100
    cv2.line(frame, tuple(avg_center[:2].astype(int)), tuple(ray_end[:2].astype(int)), (1, 174, 255), 4)
    
    cx, cy = avg_center[:2].astype(int)

    #cv2.putText(frame, f"Yaw: {yaw:.1f}", (cx + 30, cy+ 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    #cv2.putText(frame, f"Pitch: {pitch:.1f}", (cx, cy + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    draw_text_pil(frame, f"Yaw: {yaw:.0f}", (cx + 20, cy), (1, 174, 255))
    draw_text_pil(frame, f"Pitch: {pitch:.0f}", (cx- 50, cy+30), (1, 174, 255))



def draw_text_pil(frame, txt, pos, color=(0,255,0)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, txt, font=font, fill=color[::-1]) 
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame    

def calibrate(leftEye, rightEye):
    # Reset cursor center point
    calibration_yaw = 180 - raw_yaw
    calibration_pitch = 180 - raw_pitch
    print(f"[Calibrated] Offset Yaw: {calibration_yaw}, Offset Pitch: {calibration_pitch}")
    # Reset EAR thresholds
    # 0.02 is placeholder will be configurable
    leftEye.thresh = calc_thresh(leftEyePos) + 0.02
    rightEye.thresh = calc_thresh(rightEyePos) + 0.02
    


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
    
    threading.Thread(target=move_mouse, daemon=True).start()

    cap = cv2.VideoCapture(0)
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

        
        LEFT_IRIS_IDXS  = list(range(474, 479))
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in LEFT_EYE_LANDMARKS:
                    # Convert to coorinates of frame
                    x = int(landmark.x * frame.shape[1]) 
                    y = int(landmark.y * frame.shape[0]) 
                    leftEyePos.append([x,y])

                    if leftEye.closedFlag == True :
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), 10)  
                    else:
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  
    
                    
                elif idx in RIGHT_EYE_LANDMARKS:
                    x = int(landmark.x * frame.shape[1]) 
                    y = int(landmark.y * frame.shape[0]) 
                    rightEyePos.append([x,y])
                    if rightEye.closedFlag == True:
                        cv2.circle(frame, (x, y), 2, (50, 0, 255), 10)
                    else:
                        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                        
                elif idx in LEFT_IRIS_IDXS:
                    if leftEye.toggleFlag == True:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 1, (0, 255, 255), 2)                    
                    
            cursor(frame, face_landmarks)
            rightEye.checkRightEye(rightEyePos)
            leftEye.checkLeftEye(leftEyePos)
 
        
        frameCount += 1
        if (time.time() - startTime) >= 1.0:
            fps = frameCount
            startTime = time.time()
            frameCount = 0
                

        draw_text_pil(frame, f" 't' to toggle left | 'c' to calibrate | 'q' to quit", (100, 425), (1, 174, 255))
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
            calibrate(leftEye, rightEye)
            KEY_STATE['c'] = False

            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()