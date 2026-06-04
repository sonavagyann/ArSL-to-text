from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

app = Flask(__name__)

actions = np.array(['ա', 'ազգանուն', 'անուն', 'Բարև Ձեզ', 'դու', 'ես', 'երեկո', 'Երևան', 'իմ', 'ի՞նչ',
                    'լավ', 'լսել', 'Հայաստան', 'հաղթել', 'մաթեմատիկա', 'միշտ', 'ն', 'նա', 'շնորհակալություն',  'ո',
                    'ս', 'սովորել', 'սիրել', 'վատ', 'տխուր', 'ցտեսություն', 'ուսանող', 'ուտել', 'քո', 'քնել'])

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

armenian_display_map = {'Բարև Ձեզ': 'Ողջույն'}

grammar = {
    'սիրել':    {'ես': ('սիրում եմ', 'love'),    'դու': ('սիրում ես', 'love'),    'նա': ('սիրում է', 'loves')},
    'լսել':     {'ես': ('լսում եմ', 'listen'),    'դու': ('լսում ես', 'listen'),   'նա': ('լսում է', 'listens')},
    'ուտել':    {'ես': ('ուտում եմ', 'eat'),      'դու': ('ուտում ես', 'eat'),     'նա': ('ուտում է', 'eats')},
    'մտածել':   {'ես': ('մտածում եմ', 'think'),   'դու': ('մտածում ես', 'think'),  'նա': ('մտածում է', 'thinks')},
    'քնել':     {'ես': ('քնում եմ', 'sleep'),     'դու': ('քնում ես', 'sleep'),    'նա': ('քնում է', 'sleeps')},
    'սովորել':  {'ես': ('սովորում եմ', 'study'),  'դու': ('սովորում ես', 'study'), 'նա': ('սովորում է', 'studies')},
    'խնդրել':  {'ես': ('խնդրում եմ', 'study'),  'դու': ('խնդրում ես', 'study'), 'նա': ('խնդրում է', 'studies')},
    'ուսանող':  {'ես': ('ուսանող եմ', 'am a student'), 'դու': ('ուսանող ես', 'are a student'), 'նա': ('ուսանող է', 'is a student')},
}

name_surname = {
    'անուն':    {'իմ': ('անունն է', 'name is'),    'քո': ('անունն է', 'name is')},
    'ազգանուն': {'իմ': ('ազգանունն է', 'surname is'), 'քո': ('ազգանունն է', 'surname is')},
}

predicates = {
    'անուն':    ('անունը', 'name'),
    'ազգանուն': ('ազգանունը', 'surname'),
}

after_love = {
    'Հայաստան': ('Հայաստանը', 'Armenia'),
    'Երևան':    ('Երևանը', 'Yerevan'),
}

love_forms = set(v['ես'][0] for v in [grammar['սիրել']] if 'ես' in v) | \
             {grammar['սիրել'][p][0] for p in grammar['սիրել']}

model = load_model('model.h5')

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
        mp_drawing.DrawingSpec(color=(86, 13, 13),  thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(92, 16, 16),  thickness=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(86, 13, 13), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(92, 16, 16), thickness=2))

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

strict_signs = ['լավ', 'վատ']

THRESHOLD = 0.75
COOLDOWN_SECONDS = 1
STRICT_THRESHOLD = 0.85

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

            #person_detected = results.pose_landmarks is not None or results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
            person_detected = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

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

                state['all_probs']  = [round(float(p) * 100, 1) for p in res]
                state['current']    = latin_map.get(predicted, predicted)
                state['confidence'] = round(confidence * 100, 1)

                current_time = time.time()
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
        'sentence_list': [p[0] for p in pairs],
        'sentence_lat_list': [p[1] for p in pairs],
        'sentence': ' '.join([p[0] for p in pairs]),
        'current':    state['current'],
        'confidence': state['confidence'],
        'all_probs': list(zip([armenian_display_map.get(a, a) for a in actions],
                       state['all_probs'])),
        'all_probs_eng': [latin_map.get(a, a) for a in actions],
    })

@app.route('/remove/<int:index>', methods=['POST'])
def remove_word(index):
    if 0 <= index < len(state['sentence']):
        state['sentence'].pop(index)
    return jsonify({'ok': True})

def apply_grammar(sentence):
    result = []
    capitalize_next = False

    for i, word in enumerate(sentence):
        prev  = sentence[i-1] if i > 0 else None
        prev2 = sentence[i-2] if i > 1 else None
        arm = armenian_display_map.get(word, word)
        eng = latin_map.get(word, word)

        if word in grammar and prev in grammar[word]:
            arm, eng = grammar[word][prev]
            capitalize_next = False
        elif word in predicates and prev in ('իմ','քո','նրա') and prev2 == 'է':
            arm, eng = predicates[word]
            capitalize_next = True
        elif word in name_surname and prev in name_surname.get(word, {}):
            arm, eng = name_surname[word][prev]
            capitalize_next = True
        elif word in after_love and prev == 'սիրել':
            arm, eng = after_love[word]
            capitalize_next = False
        else:
            if capitalize_next:
                arm = arm[0].upper() + arm[1:] if arm else arm
                eng = eng[0].upper() + eng[1:] if eng else eng
            capitalize_next = False

        result.append((arm, eng))
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