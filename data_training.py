from pathlib import Path
import mediapipe as mp
import numpy as np
import cv2
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard

TRAIN_MODEL = False
DATA_PATH = Path('DB')

actions = np.array(['ա', 'ազգանուն', 'անուն', 'Բարև Ձեզ', 'դու', 'ես', 'երեկո', 'Երևան', 'իմ', 'ի՞նչ',
                    'լավ', 'լսել', 'Հայաստան', 'հաղթել', 'մաթեմատիկա', 'միշտ', 'ն', 'նա', 'շնորհակալություն',  'ո',
                    'ս', 'սովորել', 'սիրել', 'վատ', 'տխուր', 'ցտեսություն', 'ուսանող', 'ուտել', 'քո', 'քնել'])#-է, մտածել, անուշ, խնդրել

no_sequences = 200
sequence_length = 30

label_map = {str(label): num for num, label in enumerate(actions)}
print(label_map)
sequences, labels = [], []

print("Data retrieval started")

for action in actions:
    if action == 'իմ':
        new_action = 'իմ2'
        action_path = DATA_PATH / new_action
    elif action == 'Հայաստան':
        new_action = 'Հայաստան3'
        action_path = DATA_PATH / new_action
    elif action == 'միշտ':
        new_action = 'միշտ2'
        action_path = DATA_PATH / new_action
    elif action == 'լսել':
        new_action = 'լսել2'
        action_path = DATA_PATH / new_action
    else:
        action_path = DATA_PATH / action
        new_action = action
    for sequence_num in range(1, no_sequences + 1):
        frame_path = action_path / f"{new_action}_{sequence_num}.npy"
        #to print the loaded files
        #print(frame_path)
        res = np.load(str(frame_path), allow_pickle=False)
        sequences.append(res)
        labels.append(label_map[action])

print("Data successfully loaded")

X = np.array(sequences)
print("X.shape:", X.shape)
y = to_categorical(labels).astype(int)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("X_train.shape:", X_train.shape)
print("y_train.shape:", y_train.shape)
print("X_test.shape:", X_test.shape)
print("y_test.shape:", y_test.shape)


log_dir = Path('Logs')
log_dir.mkdir(parents=True, exist_ok=True)

tb_callback = TensorBoard(log_dir=str(log_dir))

model = Sequential()

model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 258)))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

if TRAIN_MODEL:
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback], validation_data=(X_test, y_test))#epochs were 140
    model.summary()
    model.evaluate(X_test, y_test)
    model.save('model21.h5')
else:
    from tensorflow.keras.models import load_model
    model = load_model('model21.h5')
    print("Model loaded successfully")


res = model.predict(X_test)
print(actions[np.argmax(res[5])])
print(actions[np.argmax(y_test[5])])


#Evaluation (10)
yhat = model.predict(X_test)
print("yhat:", yhat)

ytrue = np.argmax(y_test, axis=1).tolist()
print("ytrue:", ytrue)
yhat = np.argmax(yhat, axis=1).tolist()
print("yhat:", yhat)

confusion_matrix = multilabel_confusion_matrix(ytrue, yhat)
print("confusion matrix:", confusion_matrix)

accuracy = accuracy_score(ytrue, yhat)
print("accuracy:", accuracy)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

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


def collect_landmarks(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)

    return np.concatenate([pose, lh, rh])

latin_map = {
    'ա' : 'a',
    'ազգանուն' : 'surname',
    'անուն' : 'name',
    'Բարև Ձեզ' : 'Hello',
    'դու' : 'you',
    'ես' : 'I',
    'երեկո' : 'evening',
    'Երևան' : 'Yerevan',
    'իմ' : 'my',
    'ի՞նչ' : 'what',
    'լավ' : 'good',
    'լսել' : 'to listen',
    'Հայաստան' : 'Armenia',
    'հաղթել' : 'to win',
    'մաթեմատիկա' : 'mathematics',
    'միշտ' : 'always',
    'ն' : 'n',
    'նա' : 'he/she',
    'շնորհակալություն' : 'thank you',
    'ո' : 'vo',
    'ս' : 's',
    'սովորել' : 'to study',
    'սիրել' : 'to love',
    'վատ' : 'bad',
    'տխուր' : 'sad',
    'ցտեսություն' : 'good bye',
    'ուսանող' : 'student',
    'ուտել' : 'to eat',
    'քո' : 'your',
    'քնել' : 'to sleep'
}

sequence = []
sentence = []
threshold = 0.9
res = np.zeros(len(actions))
last_prediction_time = 0
COOLDOWN_SECONDS = 4

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 750)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

with mp_holistic.Holistic(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        image, results = mediapipe_detection(frame, holistic)
        #print(results)

        show_landmarks(image, results)

        keypoints = collect_landmarks(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

            current_time = time.time()
            if res[np.argmax(res)] > threshold and (current_time - last_prediction_time) > COOLDOWN_SECONDS:
                if len(sentence) == 0 or actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
                    last_prediction_time = current_time

            if len(sentence) > 5:
                sentence = sentence[-5:]

        display_text = ' '.join([latin_map.get(s, s) for s in sentence])
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, display_text, (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()