from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import dlib
from threading import Thread
import base64

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


# Initialize the dlib's face detector and facial landmark predictor
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat.bz2/shape_predictor_68_face_landmarks.dat")

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = hog_face_detector(gray)
            for face in faces:
                face_landmarks = dlib_facelandmark(gray, face)
                for n in range(0, 68):
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = base64.b64encode(buffer).decode('utf-8')
            
            yield frame


def video_stream():
    for frame in gen_frames():
        socketio.emit('frame', {'data': frame})
        socketio.sleep(0.05)  # You might adjust this sleep time

@socketio.on('start_stream')
def handle_stream():
    thread = Thread(target=video_stream)
    thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('start_stream')
def handle_stream():
    for frame in gen_frames():
        emit('frame', {'data': frame})
        socketio.sleep(0.1)  # Adjust frame rate based on your needs

if __name__ == '__main__':
    socketio.run(app, debug=True)
