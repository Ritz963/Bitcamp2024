from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from djitellopy import Tello
import cv2
import dlib
import threading
from threading import Thread
import base64
import numpy as np
from time import sleep
from math import *
import time
from socket import socket, AF_INET, SOCK_DGRAM, error as socket_error
import socket
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*')

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat.bz2/shape_predictor_68_face_landmarks.dat")

def setup_tello():
    sock = socket.socket()
    sock.bind(('', 8889))
    global tello
    print("about to init")
    tello = Tello()
    print("everything worked ")
    

def gen_frames():

    tello.connect()
    tello.streamon()
    print("Tello Battery: " + str(tello.get_battery()))

    try:
        while True:
            
            frame = cv2.resize(tello.get_frame_read().frame, (640, 480))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            RGBFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces = hog_face_detector(gray)
            score = 0
            for face in faces:
                face_landmarks = dlib_facelandmark(gray, face)

                for n in range(0, 68):
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    cv2.circle(RGBFrame, (x, y), 1, (0, 255, 255), 1)

                score = rizz_calculator(face_landmarks)
            ret, buffer = cv2.imencode('.jpg', RGBFrame)
            frame_encoded = base64.b64encode(buffer).decode('utf-8')
            yield (frame_encoded, score)
    finally:
            tello.streamoff()

def video_stream(frame_generator):
    print("Starting video stream...")
    for frame_encoded, score in frame_generator():
        print(f"Emitting score {score}")
        socketio.emit('frame', {'data': frame_encoded, 'score': score})
        socketio.sleep(0.05)

def calc_sym(f):
    return 0.5

def calc_mouth(f):
    right_mouth = f.part(54).x
    left_mouth = f.part(48).x

    right_eye = (f.part(42).x + f.part(45).x)/2
    left_eye = (f.part(36).x + f.part(39).x)/2

    diff_right = abs(right_mouth-right_eye)
    diff_left = abs(left_mouth-left_eye)
    total_diff = diff_right + diff_left
    return sig(total_diff)

def sig(x):
    return 1/(1 + np.exp(-x))

def calc_eyes(f):
    o = f.part(28).x - f.part(9).x
    a = f.part(28).y - f.part(9).y
    angle = 0
    if a != 0:
        angle = atan2(o, a)

    eye_points = [f.part(37) , f.part(40), f.part(43), f.part(46)]
    vecs = []
    for i in range(len(eye_points)):
        vecs.append(np.array([[eye_points[i].x],
                             [eye_points[i].y]]))
    rot = np.array([[cos(angle), sin(angle)],
                   [-sin(angle), cos(angle)]])
    
    for i in range(len(vecs)):
        vecs[i] = np.matmul(rot, vecs[i])
    
    score = sig(((vecs[1])[1,0]-(vecs[0])[1,0]) + ((vecs[2])[1,0]-(vecs[3])[1,0]))
    return score

def calc_jaw(f):
    return 0.5

def rizz_calculator(face):
    weights = {"mouth": 20, "eyes": 20, "jaw": 30, "sym": 30}
    return weights["mouth"]*calc_mouth(face) + weights["eyes"]*calc_eyes(face) + weights["jaw"]*calc_jaw(face) + weights["sym"]*calc_sym(face)


@socketio.on('drone_command')
def handle_drone_command(data):
    command = data['command']
    print(f"Received drone command: {command}")
    if command == 'takeoff':
        tello.takeoff()
    elif command == 'land':
        tello.land()
    elif command == 'up':
        tello.move_up(20)  # Modify as needed
    elif command == 'down':
        tello.move_down(20)
    elif command == 'left':
        tello.move_left(20)
    elif command == 'right':
        tello.move_right(20)



@socketio.on('start_stream')
def handle_stream():
    setup_tello()
    print('Starting video stream...')
    Thread(target=video_stream, args=(gen_frames,), daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')
    # Ensure the drone is safely handled on disconnect
    tello.land()
    tello.end()  # Properly end the Tello session



if __name__ == '__main__':
    socketio.run(app, debug=True)