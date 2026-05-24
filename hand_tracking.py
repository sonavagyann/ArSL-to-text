import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 750)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    success, frame = cap.read()
    if not success:
        print("There seems to be a problem")
        continue

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #to print the coordinates of the landmarks
            #print(hand_landmarks)
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 225), thickness=2)
            )

    cv2.imshow("Video frame", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()