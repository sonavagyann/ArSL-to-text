from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import time

app = Flask(__name__)

actions = np.array(['ես', 'դու', 'իմ', 'քո', 'անուն',
                    'ազգանուն', 'ի՞նչ', 'է', 'և', 'Բարև Ձեզ', 'ա', 'գ', 'դ', 'յ', 'ն', 'ո', 'ս', 'վ', 'ր', 'Ցտեսություն',
                    'նա', 'նրա', 'մենք', 'սիրել', 'լսել', 'ժեստերի լեզու', 'շնորհակալություն', 'այո', 'վատ', 'լավ'])

latin_map = {
    'ես': 'I',
    'դու': 'you',
    'իմ': 'my',
    'քո': 'your',
    'անուն': 'name',
    'ազգանուն': 'surname',
    'ի՞նչ': 'what',
    'է': 'is',
    'և': 'and',
    'Բարև Ձեզ': 'Hello',
    'ա': 'a',
    'գ' : 'g',
    'դ' : 'd',
    'յ' : 'y',
    'ն': 'n',
    'ո': 'vo',
    'ս': 's',
    'վ' : 'v',
    'ր' : 'r',
    'Ցտեսություն' : 'Good bye',
    'նա' : 'he/she',
    'նրա' : 'his/her',
    'մենք' : 'we',
    'սիրել' : 'love',
    'լսել' : 'listen',
    'ժեստերի լեզու' : 'sign language',
    'շնորհակալություն' : 'thank you',
    'այո' : 'yes',
    'վատ' : 'bad',
    'լավ' : 'good'
}

armenian_display_map = {
    'Բարև Ձեզ': 'Ողջույն',
}

english_verb_map = {
    'սիրել': 'love',
}

verb_conjugations = {
    'սիրել': {
        'ես': 'սիրում եմ',
        'դու': 'սիրում ես',
        'նա': 'սիրում է',
        'մենք': 'սիրում ենք'
    },
    'լսել': {
        'ես': 'լսում եմ',
        'դու': 'լսում ես',
        'նա': 'լսում է',
        'մենք': 'լսում ենք'
    }
}

model = load_model('model12.h5')

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

def mediapipe_detection(image, holistic_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def show_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(3, 0, 46),    thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(1, 0, 72),    thickness=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(86, 13, 13),  thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(92, 16, 16),  thickness=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 165, 255), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(0, 165, 255), thickness=2))

def collect_landmarks(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(132)
    lh   = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
           if results.left_hand_landmarks else np.zeros(63)
    rh   = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
           if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

state = {
    'sentence':   [],
    'current':    '',
    'confidence': 0.0,
    'all_probs':  [0.0] * len(actions),
    'last_pred_time': 0,
    'prediction_counter': {}
}

strict_signs = ['սիրել', 'այո', 'ժեստերի լեզու']

THRESHOLD = 0.75
COOLDOWN_SECONDS = 5
STRICT_THRESHOLD = 0.9

def gen_frames():
    sequence = []
    frame_count = 0
    recent_predictions = []
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 750)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

    with mp_holistic.Holistic(model_complexity=1,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            show_landmarks(image, results)

            keypoints = collect_landmarks(results)
            sequence.append(keypoints)
            sequence[:] = sequence[-30:]

            person_detected = results.pose_landmarks is not None or results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
            #person_detected = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

            if not person_detected:
                state['all_probs'] = [0.0] * len(actions)
                state['confidence'] = 0.0
                state['current'] = ''
                recent_predictions.clear()
                sequence.clear()
            elif frame_count % 5 == 0 and len(sequence) == 30:
                res        = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

                if len(recent_predictions) > 1:
                    prev_idx = int(np.argmax(recent_predictions[-1]))
                    curr_idx = int(np.argmax(res))
                    if prev_idx != curr_idx:
                        recent_predictions = [res]

                recent_predictions.append(res)
                recent_predictions = recent_predictions[-2:]
                smoothed_res = np.mean(recent_predictions, axis=0)

                idx = int(np.argmax(smoothed_res))
                confidence = float(smoothed_res[idx])
                predicted = actions[idx]

                # idx        = int(np.argmax(res))
                # confidence = float(res[idx])
                # predicted  = actions[idx]


                #state['all_probs'] = [round(float(p) * 100, 1) for p in smoothed_res]
                state['all_probs']  = [round(float(p) * 100, 1) for p in res]
                state['current']    = latin_map.get(predicted, predicted)
                state['confidence'] = round(confidence * 100, 1)

                current_time = time.time()
                #if confidence > THRESHOLD and (current_time - state['last_pred_time']) > COOLDOWN_SECONDS:
                current_threshold = STRICT_THRESHOLD if predicted in strict_signs else THRESHOLD
                if confidence > current_threshold:
                    state['prediction_counter'][predicted] = state['prediction_counter'].get(predicted, 0) + 1
                    if state['prediction_counter'][predicted] >= 3 and (current_time - state['last_pred_time']) > COOLDOWN_SECONDS:
                        if not state['sentence'] or predicted != state['sentence'][-1]:
                            state['sentence'].append(predicted)
                        if len(state['sentence']) > 50:
                            state['sentence'] = state['sentence'][-50:]
                        state['last_pred_time'] = current_time
                        state['prediction_counter'] = {}
                else:
                    state['prediction_counter'] = {}

            _, buffer = cv2.imencode('.jpg', image)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    sign_list = [{'arm': armenian_display_map.get(a, a), 'lat': latin_map.get(a, a)} for a in actions]
    return render_template('index.html', signs=sign_list)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    pairs = apply_grammar(state['sentence'])
    return jsonify({
        #'sentence': ' '.join([armenian_display_map.get(w, w) for w in state['sentence']]),
        'sentence_list': [p[0] for p in pairs],
        'sentence_lat_list': [p[1] for p in pairs],
        'sentence': ' '.join([p[0] for p in pairs]),
        #'sentence': ' '.join(apply_grammar([armenian_display_map.get(w, w) for w in state['sentence']])),
        # 'sentence_lat': ' '.join([latin_map.get(armenian_display_map.get(w, w), latin_map.get(w, w)) for w in state['sentence']]),
        #'sentence_lat_list': [latin_map.get(armenian_display_map.get(w, w), latin_map.get(w, w)) for w in state['sentence']],
        'current':    state['current'],
        'confidence': state['confidence'],
        'all_probs': list(zip([armenian_display_map.get(a, a) for a in actions],
                       state['all_probs'])),
    })

def apply_grammar(sentence):
    result = []

    for i, word in enumerate(sentence):
        if word in verb_conjugations and i > 0:
            prev_word = sentence[i - 1]
            conjugated = verb_conjugations[word].get(prev_word, word)
            english = english_verb_map.get(word, latin_map.get(word, word))
            result.append((conjugated, english))
        else:
            arm = armenian_display_map.get(word, word)
            english = latin_map.get(arm, latin_map.get(word, word))
            result.append((arm, english))
    return result

@app.route('/clear', methods=['POST'])
def clear():
    state['sentence']   = []
    state['current']    = ''
    state['confidence'] = 0.0
    state['last_pred_time'] = 0
    state['prediction_counter'] = {}
    return jsonify({'ok': True})

if __name__ == '__main__':
    app.run(debug=False)