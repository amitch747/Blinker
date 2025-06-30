import cv2
import mediapipe as mp

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
    
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    eyePoints = []
    if results.multi_face_landmarks:
        # face_landmarks look like this:
                #   NormalizedLandmark #0:
                #   x: 0.5971359014511108
                #   y: 0.485361784696579
                #   z: -0.038440968841314316
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):

                # Example: Draw a small circle at landmark index 1 (nose tip)
                if idx in RIGHT_EYE_INDICES:
                    x = int(landmark.x * frame.shape[1])  # Convert to coords of frame
                    y = int(landmark.y * frame.shape[0]) 
                    eyePoints.append([x,y])
                    cv2.circle(frame_bgr, (x, y), 2, (0, 255, 0), -1)  
        
        print(eyePoints)
    cv2.imshow('Mediapipe Face Mesh', frame_bgr)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# RIGHT EYE
# p1 = 33
# p2 = 160
# p3 = 158
# p4 = 133
# p5 = 153
# p6 = 144