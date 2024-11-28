import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from LEAP_Hand_API.python.leap_hand_utils.dynamixel_client import *
import LEAP_Hand_API.python.leap_hand_utils.leap_hand_utils as lhu
import time

# Function to display text on the image
def display_text(img, text, position, color):
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

class LeapNode:
    def __init__(self):
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        
        # Motor setup
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, 'COM13', 4000000)
                self.dxl_client.connect()
        
        # Control parameters
        # self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        # self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2)
        # self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2)
        # self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2)
        # self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2)
        # self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2)
        # self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        # self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def get_counter_move(self, player_move):
        if player_move == "Rock":
            return "Paper"
        elif player_move == "Paper":
            return "Scissors"
        elif player_move == "Scissors":
            return "Rock"

    def perform_move(self, move):
        moves = {
            "Rock": np.array([3.1416, 4.1888, 4.5553, 4.4157, 3.1416, 4.1190, 5.1487, 4.2412, 3.1416, 4.2237, 4.7124, 4.4506, 2.6005, 1.5184, 4.6775, 4.4157]),       
            "Paper": np.array([3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416]),     
            "Scissors": np.array([3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 4.2237, 4.7124, 4.4506, 2.6005, 1.5184, 4.6775, 4.4157]) 
        }
        if move in moves:
            pose = moves[move]
            # self.set_allegro(pose)  
            self.dxl_client.write_desired_pos(self.motors, pose)
            print(f"LEAP Hand performs: {move}")
            print(pose)
        else:
            print("Unknown move")

# Integrate vision-based gesture detection with LEAP hand control
def play_rockpaperscissor(leap_hand):
    # Initialize camera and hand detector
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    detector = HandDetector(maxHands=1)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame.")
            break

        # Detect hand and determine gesture
        hands, img = detector.findHands(img)
        gesture = "Invalid Gesture"
        if hands:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            if fingers == [0, 0, 0, 0, 0]:
                gesture = "Rock"
            elif fingers == [1, 1, 1, 1, 1]:
                gesture = "Paper"
            elif fingers == [0, 1, 1, 0, 0]:
                gesture = "Scissors"

            # if gesture in ["Rock", "Paper", "Scissors"]:
            counter_move = leap_hand.get_counter_move(gesture)
            print(f"Player move: {gesture}")
            print(f"LEAP Hand counter move: {counter_move}")
            leap_hand.perform_move(counter_move)

            # Display detected gesture
            display_text(img, f"Gesture: {gesture}", (10, 70), (255, 0, 0))
        
        # Show image with gesture
        cv2.imshow("Rock-Paper-Scissors", img)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    leap_hand = LeapNode()
    play_rockpaperscissor(leap_hand)
    
if __name__ == "__main__":
    main()