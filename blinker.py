import cv2
import mediapipe as mp
import pyautogui
import time
import math
import socket

from ear import calc_EAR
from utils import calc_3D_dist

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1', 54000))


NOSE_LANDMARK = 1
LEFT_EYE_LANDMARKS = [362, 380, 373, 263, 387, 385]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYEBROW_LANDMARKS = [63, 105, 66, 107, 65, 52, 53]

neutral_x = None
neutral_y = None








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
class Brow:
    # Needs a massive rework. Brow height is very inconsitent right now
    def __init__(self):
        self.upFlag = False
        self.downFlag = False
    # THIS WILL NOT WORK. SCALE IS NOT LINEAR FOR POINTS FURTHER BACK!
        #print("Brow Height: ", str(height / (points[0]-points[1])), "%")
    def checkBrowHeight(self, percent):
        #print(self.upFlag)          
        if (percent > 89 and not self.upFlag) :
            self.upFlag = True
            self.downFlag = False
            pyautogui.press('space')
        elif (percent < 86 and not self.downFlag):
            self.downFlag = True
            self.upFlag = False
            pyautogui.keyDown('ctrl')          
        elif (86  < percent < 89):
            if (self.downFlag):
                pyautogui.keyUp('ctrl')          
            self.downFlag = False
            self.upFlag = False
                 


  
def cursor(frame_bgr,face_landmarks):
    DEADZONE = 10   # Pixels
    EDGE_MARGIN = 50   # Pixels from screen edge to stop moving
    
    global neutral_x, neutral_y

    frame_height, frame_width = frame_bgr.shape[:2]

    lm = face_landmarks.landmark[NOSE_LANDMARK]

    x = lm.x * frame_width
    y = lm.y * frame_height

    cv2.circle(frame_bgr, (int(x), int(y)), 4, (0, 255, 0), -1)

    if neutral_x is None or neutral_y is None:
        neutral_x = x
        neutral_y = y
        print(f"Calibrated neutral at x={neutral_x:.2f}, y={neutral_y:.2f}")
    else:
        dx = x - neutral_x
        dy = y - neutral_y


        cv2.circle(frame_bgr, (int(neutral_x), int(neutral_y)), int(DEADZONE), (0, 0, 255), 1)
        cv2.circle(frame_bgr, (int(neutral_x), int(neutral_y)), int(DEADZONE*3), (0, 0, 255), 1)
        cv2.circle(frame_bgr, (int(neutral_x), int(neutral_y)), int(DEADZONE*6), (0, 0, 255), 1)

        vec = math.sqrt(abs(dx)**2 + abs(dy)**2 ) 

        if (vec < DEADZONE*3):
            SENSITIVITY = 3.0
        elif (DEADZONE*3 < vec < DEADZONE*6) :
            SENSITIVITY = 4.0
        elif (vec > DEADZONE*6):
            SENSITIVITY = 5.0
        
        cv2.putText(frame_bgr, "Sens:"+str(SENSITIVITY), (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame_bgr, "Vec:"+str(vec), (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # Apply deadzone
        if abs(dx) < DEADZONE:
            dx = 0
        if abs(dy) < DEADZONE:
            dy = 0

        # Apply sensitivity
        move_x = -int(dx * SENSITIVITY)
        move_y = -int(-dy * SENSITIVITY)  # Invert Y so down head = down mouse
        
        cv2.circle(frame_bgr, (int(neutral_x), int(neutral_y)), 4, (0, 0, 255), -1)
        
        cv2.line(frame_bgr, (int(neutral_x), int(neutral_y)), (int(x), int(y)), (0, 255, 0), 2)

        # # Edge check: prevent fail-safe
        screen_width, screen_height = pyautogui.size()
        mouse_x, mouse_y = pyautogui.position()
            
        # message = f"{mouse_x} {mouse_y}"
        # client.sendall(message.encode())
        
        if move_x != 0 or move_y != 0:
            message = f"{move_x} {move_y}"
            client.sendall(message.encode())
        
        if (EDGE_MARGIN < mouse_x < screen_width - EDGE_MARGIN and
            EDGE_MARGIN < mouse_y < screen_height - EDGE_MARGIN):
            if move_x != 0 or move_y != 0:
                pyautogui.moveRel(move_x, move_y)


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
    brow = Brow()
    
    frameCount = 0  
    fps = 0
    startTime = time.time()
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failure to read from camera")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        # Returns a NamedTuple object with a "multi_face_landmarks" field 
        # that contains the face landmarks on each detected face.
        results = face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        leftEyePos = []
        rightEyePos = []
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
        
            cursor(frame_bgr, face_landmarks)

            
            browMark = face_landmarks.landmark[105] 
            chinMark = face_landmarks.landmark[152]

            browToChin = calc_3D_dist(browMark, chinMark)
            faceHeight = calc_3D_dist(face_landmarks.landmark[10], chinMark)  
            
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in LEFT_EYE_LANDMARKS:
                    # Convert to coorinates of frame
                    x = int(landmark.x * frame.shape[1]) 
                    y = int(landmark.y * frame.shape[0]) 
                    leftEyePos.append([x,y])
                    cv2.circle(frame_bgr, (x, y), 2, (0, 255, 0), -1)  
                elif idx in RIGHT_EYE_LANDMARKS:
                    x = int(landmark.x * frame.shape[1]) 
                    y = int(landmark.y * frame.shape[0]) 
                    rightEyePos.append([x,y])
                    cv2.circle(frame_bgr, (x, y), 2, (0, 0, 255), -1)
                elif idx in RIGHT_EYEBROW_LANDMARKS:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame_bgr, (x, y), 2, (255, 0, 0), -1)

                    

        cv2.putText(frame_bgr, f"Face height: {faceHeight}", (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        percent = (browToChin / faceHeight) * 100
        cv2.putText(frame_bgr, f"Brow to chin: {percent}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)


        rightEye.checkRightEye(rightEyePos)
        leftEye.checkLeftEye(leftEyePos)
        brow.checkBrowHeight(percent) 

    
        frameCount += 1
        if (time.time() - startTime) >= 1.0:
            fps = frameCount
            startTime = time.time()
            frameCount = 0
            


        cv2.putText(frame_bgr, "'c' - recalibrate | 't' - toggle left | 'q' - quit", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame_bgr, f"FPS: {fps}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame_bgr, f"Left Toggle: {leftEye.toggleFlag}", (10,60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame_bgr, f"Left: {leftEye.closedFlag}, Right: {rightEye.closedFlag}", (10,300), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.putText(frame_bgr, f"Jump: {brow.upFlag}, Crouch: {brow.downFlag}", (10,320), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)


        cv2.imshow('Blinker', frame_bgr)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break 
        if key == ord('t'):
            leftEye.toggle()
        if key == ord('c'):
            global neutral_x, neutral_y            
            neutral_x = None
            neutral_y = None
            print("Recalibrated neutral point.")
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()