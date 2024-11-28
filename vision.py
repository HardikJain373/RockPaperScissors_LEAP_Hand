import cv2
from cvzone.HandTrackingModule import HandDetector

# Function to display text on the image
def display_text(img, text, position, color):
    cv2.putText(
        img,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera.")
    exit()

detector = HandDetector(maxHands=2)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame.")
        break
    img = cv2.flip(img, 1)

    # Detect hand and find gesture
    hands, img = detector.findHands(img)  # with draw

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)

        # Detect hand gesture
        if fingers == [0, 0, 0, 0, 0]:
            gesture = "Rock"
        elif fingers == [1, 1, 1, 1, 1]:
            gesture = "Paper"
        elif fingers == [0, 1, 1, 0, 0]:
            gesture = "Scissors"
        else:
            gesture = "Invalid Gesture"

        # Display the detected gesture
        display_text(img, f"Gesture: {gesture}", (10, 70), (255, 0, 0))
        print(fingers)

    # Show the image with gesture
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):  # Exit when 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()