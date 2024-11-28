# #!/usr/bin/env python3
# import cv2
# import numpy as np
# import tensorflow as tf
# import pickle
# import time
# from leap_hand_utils.dynamixel_client import *
# import leap_hand_utils.leap_hand_utils as lhu

# def perform_action(gesture, leap_hand_control):
#     if gesture == 'stone':
#         rock = [
#             173.5, 227.3, 321.2, 180.0, 178.0, 285.1, 244.8, 228.3,
#             176.3, 280.9, 263.4, 213.1, 183.4, 90.2, 252.6, 290.8
#         ]
#         leap_hand_control.move_to_positions_slowly(rock, steps=200, delay=0.005)
#         print("Performed 'rock' action.")
#     elif gesture == 'paper':
#         paper = [
#             174.2, 182.5, 180.88, 179.91, 177.1, 191.87, 172.18, 189.32,
#             177.89, 194.33, 162.51, 193.1, 176.75, 85.25, 183.25, 186.24
#         ]
#         leap_hand_control.move_to_positions_slowly(paper, steps=200, delay=0.005)
#         print("Performed 'paper' action.")
#     elif gesture == 'scissor':
#         scissor = [
#             173.5, 184.5, 180.88, 180.0, 177.7, 192.3, 172.4, 189.6,
#             176.3, 280.9, 263.4, 213.1, 183.4, 90.2, 252.6, 290.7
#         ]
#         leap_hand_control.move_to_positions_slowly(scissor, steps=200, delay=0.005)
#         print("Performed 'scissor' action.")

# def recognize_gesture(leap_hand_control):
#     # Load the trained model
#     model = tf.keras.models.load_model('gesture_model.h5')
    
#     # Load the label encoder
#     with open('label_encoder.pkl', 'rb') as f:
#         le = pickle.load(f)
    
#     cap = cv2.VideoCapture(0)
#     recognized_gesture = None  # Variable to store the recognized gesture

#     while True:
#         ret, frame = cap.read()
#         if ret:
#             frame = cv2.flip(frame, 1)  # Flip frame to avoid mirror effect
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             gray = cv2.resize(gray, (64, 64))
#             gray = np.expand_dims(gray, axis=-1)
#             gray = gray / 255.0  # Normalize pixel values
#             gray = np.expand_dims(gray, axis=0)

#             # Predict the gesture
#             prediction = model.predict(gray)
#             recognized_gesture = le.inverse_transform([np.argmax(prediction)])[0]

#             # Display the result
#             cv2.putText(frame, f'Gesture: {recognized_gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#             cv2.imshow('Gesture Recognition', frame)

#             # Perform action based on the recognized gesture
#             perform_action(recognized_gesture, leap_hand_control)

#             # Print the recognized gesture
#             print(f'Recognized Gesture: {recognized_gesture}')

#             # Press 'q' to quit
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()
#     return recognized_gesture

# class LeapHandControl:
#     def _init_(self):
#         # Dynamixel motor parameters
#         self.motors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        
#         # Try connecting to the motor controller on available ports
#         try:
#             self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB0', 4000000)
#             self.dxl_client.connect()
#         except Exception:
#             try:
#                 self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB1', 4000000)
#                 self.dxl_client.connect()
#             except Exception:
#                 self.dxl_client = DynamixelClient(self.motors, 'COM13', 4000000)
#                 self.dxl_client.connect()
        
#         # Enable torque and set position control
#         self.dxl_client.set_torque_enabled(self.motors, True)

#     def move_to_positions_slowly(self, target_positions_degrees, steps=100, delay=0.05):
#         # Convert target positions from degrees to radians
#         target_positions = np.radians(target_positions_degrees)
        
#         # Get the current positions to start from
#         current_positions = np.array(self.dxl_client.read_pos())
        
#         # Compute incremental step sizes for each joint
#         step_sizes = (target_positions - current_positions) / steps
        
#         # Gradually move to the target position
#         for step in range(steps):
#             current_positions += step_sizes
#             self.dxl_client.write_desired_pos(self.motors, current_positions)
#             time.sleep(delay)  # Wait a little between steps for slow movement
#         print("Reached target positions.")

# if _name_ == "_main_":
#     # Initialize the LeapHandControl
#     leap_hand_control = LeapHandControl()
    
#     # Run real-time gesture recognition and print the result
#     gesture = recognize_gesture(leap_hand_control)
#     print(f'Final Recognized Gesture: {gesture}')

#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
import time
from LEAP_Hand_API.python.leap_hand_utils.dynamixel_client import *
import LEAP_Hand_API.python.leap_hand_utils.leap_hand_utils as lhu

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def recognize_gesture(landmarks):
    # Get coordinates of necessary landmarks
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    
    # Define simple rules for rock, paper, scissors
    if thumb_tip.y < index_tip.y:
        return "stone"
    elif thumb_tip.y > pinky_tip.y:
        return "paper"
    else:
        return "scissor"

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            return recognize_gesture(landmarks)
    return None

rock = [
    173.5, 227.3, 321.2, 180.0, 178.0, 285.1, 244.8, 228.3,
    176.3, 280.9, 263.4, 213.1, 183.4, 90.2, 252.6, 290.8
]

scissor = [
    173.5, 184.5, 180.88, 180.0, 177.7, 192.3, 172.4, 189.6,
    176.3, 280.9, 263.4, 213.1, 183.4, 90.2, 252.6, 290.7
]

paper = [
    174.2, 182.5, 180.88, 179.91, 177.1, 191.87, 172.18, 189.32,
    177.89, 194.33, 162.51, 193.1, 176.75, 85.25, 183.25, 186.24
]

def perform_action(gesture, leap_hand_control):
    if gesture == 'stone':
        leap_hand_control.move_to_positions_slowly(paper, steps=10, delay=0.005)
        print("Performed 'paper' action.")
    elif gesture == 'paper':
        leap_hand_control.move_to_positions_slowly(scissor, steps=10, delay=0.005)
        print("Performed 'scissor' action.")
    elif gesture == 'scissor':
        leap_hand_control.move_to_positions_slowly(rock, steps=10, delay=0.005)
        print("Performed 'rock' action.")

class LeapHandControl:
    def _init_(self):
        # Dynamixel motor parameters
        self.motors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        
        # Try connecting to the motor controller on available ports
        try:
            self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(self.motors, 'COM13', 4000000)
                self.dxl_client.connect()
        
        # Enable torque and set position control
        self.dxl_client.set_torque_enabled(self.motors, True)

    def move_to_positions_slowly(self, target_positions_degrees, steps=1, delay=0.05):
        # Convert target positions from degrees to radians
        target_positions = np.radians(target_positions_degrees)
        
        # Get the current positions to start from
        current_positions = np.array(self.dxl_client.read_pos())
        
        # Compute incremental step sizes for each joint
        step_sizes = (target_positions - current_positions) / steps
        
        # Gradually move to the target position
        for step in range(steps):
            current_positions += step_sizes
            self.dxl_client.write_desired_pos(self.motors, current_positions)
            time.sleep(delay)  # Wait a little between steps for slow movement
        print("Reached target positions.")

if __name__ == "_main_":
    # Initialize the LeapHandControl
    leap_hand_control = LeapHandControl()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gesture = process_frame(frame)

        if gesture:
            print(f'Gesture: {gesture}')
            cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Perform the action corresponding to the gesture
            perform_action(gesture, leap_hand_control)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()