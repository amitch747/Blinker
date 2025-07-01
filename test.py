import cv2
import mediapipe as mp
from ear import CheckEAR

import pyautogui
import time
import math







frameCount = 0
startTime = time.time()
fps = 0

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
        if self.toggleFlag:
            if CheckEAR(leftEyePos, 0.8):
                pyautogui.click()
                print('left click (tog)')
            return

        if CheckEAR(leftEyePos, 0.75):
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
        if CheckEAR(rightEyePos, 0.8):
            if not self.closedFlag and (time.time() - self.lastClick) > self.coolDown:
                pyautogui.rightClick()
                self.closedFlag = True
                self.lastClick = time.time()
                print('right click')
        else:
            self.closedFlag = False



leftEye = LeftEye()
rightEye = RightEye()

# Init mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
""" 
    FaceMesh Args:
        - static_image_mode: Whether to treat the input images as a batch of static
        and possibly unrelated images, or a video stream.
        - max_num_faces: Maximum number of faces to detect. 
        refine_landmarks: Whether to further refine the landmark coordinates
        around the eyes and lips, and output additional landmarks around the
        irises. Default to False. S
        - min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
        detection to be considered successful. 
        - min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
        face landmarks to be considered tracked successfully.
"""

# Need these for drawing
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 0 should be webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Unable to read from camera")
        continue


    # OpenCV loves BGR for some reason
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False

    # Returns a NamedTuple object with a "multi_face_landmarks" field that contains the
    # face landmarks on each detected face.
    results = face_mesh.process(frame_rgb)

    frame_rgb.flags.writeable = True
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    LEFT_EYE_POINTS = [362, 380, 373, 263, 387, 385]
    RIGHT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
    leftEyePos = []
    rightEyePos = []
    
    RIGHT_EYEBROW_INDICES = [63, 105, 66, 107, 65, 52, 53]

    rightBrowPos = []
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):

                # Example: Draw a small circle at landmark index 1 (nose tip)
                if idx in LEFT_EYE_POINTS:
                    x = int(landmark.x * frame.shape[1])  # Convert to coords of frame
                    y = int(landmark.y * frame.shape[0]) 
                    leftEyePos.append([x,y])
                    cv2.circle(frame_bgr, (x, y), 2, (0, 255, 0), -1)  
                elif idx in RIGHT_EYE_POINTS:
                    x = int(landmark.x * frame.shape[1])  # Convert to coords of frame
                    y = int(landmark.y * frame.shape[0]) 
                    rightEyePos.append([x,y])
                    cv2.circle(frame_bgr, (x, y), 2, (0, 0, 255), -1)

                elif idx in RIGHT_EYEBROW_INDICES:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    rightBrowPos.append(y)
                    cv2.circle(frame_bgr, (x, y), 2, (255, 0, 0), -1)

                

        rightEye.checkRightEye(rightEyePos)
        leftEye.checkLeftEye(leftEyePos)
        

   
    frameCount += 1
    if (time.time() - startTime) >= 1.0:
        fps = frameCount
        startTime = time.time()
        frameCount = 0
        
    

   
    cv2.putText(frame_bgr, f"FPS: {fps}", (10, 45), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.putText(frame_bgr, f"Left Toggle: {leftEye.toggleFlag}", (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    # add all stats here
    cv2.imshow('Blinker', frame_bgr)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('t'):
        leftEye.toggle()
        #left click toggle (no cooldown)

cap.release()
cv2.destroyAllWindows()




# LEFT EYE
# p1 = 33
# p2 = 160
# p3 = 158
# p4 = 133
# p5 = 153
# p6 = 144


# RIGHT EYE
# p1 = 362
# p2 = 380
# p3 = 374
# p4 = 263
# p5 = 386
# p6 = 385

