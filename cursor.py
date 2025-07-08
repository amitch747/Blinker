import numpy as np
import cv2
import math
from collections import deque
import threading
import pyautogui

from utils import draw_text_pil
from settings import DEQUE_FRAMES, DEFAULT_SENS, BOOSTED_SENS, DEADZONE, BOOSTZONE


MOUSE_MUTEX = threading.Lock()
WIDTH, HEIGHT = pyautogui.size()
SCREEN_CENTER_X = WIDTH // 2
SCREEN_CENTER_Y = HEIGHT // 2
MOUSE_LOCATION = [SCREEN_CENTER_X, SCREEN_CENTER_Y]

FACE_LANDMARKS = {1:"NOSE",10:"TOP",152:"BOTTOM",234:"LEFT",454:"RIGHT"}
deque_len = DEQUE_FRAMES
face_center_queue = deque(maxlen=deque_len)
out_vec_queue = deque(maxlen=deque_len)


calibration_yaw = 0
calibration_pitch = 0
raw_yaw = 180
raw_pitch = 180

sensitivity = 0
default_sensitivity = DEFAULT_SENS
boosted_sensitivity = BOOSTED_SENS

def move_mouse(client):
    while True:
        with MOUSE_MUTEX:
            x, y = MOUSE_LOCATION
            dx = int((x - SCREEN_CENTER_X) * sensitivity)
            dy = int((y - SCREEN_CENTER_Y) * sensitivity)
            # Reset the position to center after calculating delta
            MOUSE_LOCATION[0] = SCREEN_CENTER_X
            MOUSE_LOCATION[1] = SCREEN_CENTER_Y
        client.sendall(f"{dx} {dy}\n".encode())



def calibrate_cursor():
    global calibration_yaw, calibration_pitch
    # Reset cursor center point
    calibration_yaw = 180 - raw_yaw
    calibration_pitch = 180 - raw_pitch
    print(f"[Calibrated] Offset Yaw: {calibration_yaw}, Offset Pitch: {calibration_pitch}")

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
    
    if y_target > HEIGHT - 10:
        y_target = HEIGHT - 10
    elif y_target < 10:
        y_target = 10
    if x_target >  WIDTH-10:
        x_target = WIDTH - 10
    elif x_target < 10:
        x_target = 10
        
    #print(f"Screen position: x={x_target}, y={y_target}")

    deadzone_radius = DEADZONE / 2
    sens_boost_radius = BOOSTZONE / 2
    global sensitivity

    # Fill array
    with MOUSE_MUTEX:
        if ((pitch < 180 - sens_boost_radius or pitch > 180 + sens_boost_radius) or (yaw < 180 - sens_boost_radius or yaw > 180 + sens_boost_radius)):
            sensitivity = boosted_sensitivity
        else:
            sensitivity = default_sensitivity
            
        if  (180 - deadzone_radius < yaw < 180 + deadzone_radius):
            MOUSE_LOCATION[0] = SCREEN_CENTER_X
        else:       
            MOUSE_LOCATION[0] = x_target      
        if (180 - deadzone_radius < pitch < 180 + deadzone_radius):
            MOUSE_LOCATION[1] = SCREEN_CENTER_Y
        else:
            MOUSE_LOCATION[1] = y_target
        

    draw_text_pil(frame, f"Sens: {sensitivity}", (500, 40), (1, 174, 255))
   
    # 100 is arbitrary, should be configurable
    ray_end = avg_center - avg_out_unit * 100
    cv2.line(frame, tuple(avg_center[:2].astype(int)), tuple(ray_end[:2].astype(int)), (1, 174, 255), 4)
    
    cv2.circle(frame, tuple(avg_center[:2].astype(int)), int(DEADZONE), (0, 0, 128), 1)
    cv2.circle(frame, tuple(avg_center[:2].astype(int)), int(BOOSTZONE), (0, 0, 255), 1)


    cx, cy = avg_center[:2].astype(int)

    #cv2.putText(frame, f"Yaw: {yaw:.1f}", (cx + 30, cy+ 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    #cv2.putText(frame, f"Pitch: {pitch:.1f}", (cx, cy + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    draw_text_pil(frame, f"Yaw: {yaw:.0f}", (cx + 20, cy), (1, 174, 255))
    draw_text_pil(frame, f"Pitch: {pitch:.0f}", (cx- 50, cy+30), (1, 174, 255))

