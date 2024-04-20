from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import dlib

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize Dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this path is correct

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Detect and process facial landmarks
            frame = detect_landmarks(frame)
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            # Convert to base64 and emit via SocketIO
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_frame', {'data': 'data:image/jpeg;base64,' + frame_base64})

def detect_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    print('Client connected')
    gen_frames()

if __name__ == '__main__':
    socketio.run(app, debug=True)
