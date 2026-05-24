import cv2
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
from pathlib import Path

# 200 samples: ես, դու, իմ, քո, անուն, ազգանուն, ի՞նչ, է, և,
# ա, ս, ո, ն, ցտեսություն, վ, դ, գ, յ, ր
# նա, նրա, մենք, սիրել, լսել, ժեստերի լեզու, շնորհակալություն, այո, վատ, լավ
# next in line: ե, ներողություն, ապրել, հիշել,
CURRENT_ACTION = 'լավ'
DATA_PATH = Path('DB')
no_sequences = 200
sequence_length = 30

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 750)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


#shows the pose, and hands landmarks
def show_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(3, 0, 46), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(1, 0, 72), thickness=2)
                              )#blue

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(86, 13, 13), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(92, 16, 16), thickness=2)
                              )#red

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 165, 255), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(0, 165, 255), thickness=2)
                              )#orange



#collects the pose, left and right hand landmarks for each sign
def collect_landmarks(results):
    #pose landmarks
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    #left hand landmarks
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    #right hand landmarks
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)

    return np.concatenate([pose, lh, rh])



action_folder = DATA_PATH / CURRENT_ACTION
action_folder.mkdir(parents=True, exist_ok=True)
#2 for better accuracy, 1 for better speed
with mp_holistic.Holistic(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # while True:
    #     success, frame = cap.read()
    #     if not success:
    #         break
    #
    #     frame = cv2.flip(frame, 1)
    #
    #     image, results = mediapipe_detection(frame, holistic)
    #     print(results)
    #
    #     show_landmarks(image, results)
    #
    #     cv2.imshow("Video frame", image)
    #
    #     if cv2.waitKey(5) & 0xFF == ord('q'):
    #         break

    for sequence in range(1, no_sequences + 1):
        # We will store 30 frames worth of landmarks here
        this_sequence_landmarks = []

        for frame_num in range(sequence_length):
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            show_landmarks(image, results)

            #Recording part
            if frame_num == 0:
                cv2.putText(image, 'STARTING', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f'Collecting frames for {CURRENT_ACTION} Video No. {sequence}', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Video Feed', image)
                cv2.waitKey(2000)
            else:
                cv2.putText(image, f'Collecting frames for {CURRENT_ACTION} Video No. {sequence}', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Video Feed', image)

            keypoints = collect_landmarks(results)
            this_sequence_landmarks.append(keypoints)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # --- SAVE THE ARRAY ---
        # Convert list to numpy array and save
        file_path = action_folder / f"{CURRENT_ACTION}_{sequence}.npy"
        np.save(file_path, np.array(this_sequence_landmarks), allow_pickle=False)
        print(f"Saved {file_path}")

    cap.release()
    cv2.destroyAllWindows()


show_landmarks(frame, results)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
