import cv2
import dlib
from djitellopy import Tello
from random import randrange
import pygame
from time import sleep
from time import perf_counter
import numpy as np
from math import *

cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat.bz2/shape_predictor_68_face_landmarks.dat")

def calc_sym(f):
    return 0.5

def calc_mouth(f):
    right_mouth = face_landmarks.part(54).x
    left_mouth = face_landmarks.part(48).x

    right_eye = (face_landmarks.part(42).x + face_landmarks.part(45).x)/2
    left_eye = (face_landmarks.part(36).x + face_landmarks.part(39).x)/2

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

def get_key_press(key):
    ans = False
    for _ in pygame.event.get(): 
        pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(key))
    print('K_{}'.format(key))

    if keyInput[myKey]:
        ans = True

    pygame.display.update()
    return ans

def process_key_in(tello):
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 30
    if get_key_press("LEFT"):
        lr = -speed
    elif get_key_press("RIGHT"):
        lr = speed
    if get_key_press("UP"):
        fb = speed
    elif get_key_press("DOWN"):
        fb = -speed
    if get_key_press("w"):
        ud = speed
    elif get_key_press("s"):
        ud = -speed
    if get_key_press("a"):
        yv = -speed
    elif get_key_press("d"):
        yv = speed
    if get_key_press("q"):
        tello.land()
        sleep(3)
    if get_key_press("e"):
        tello.takeoff()
    return [lr, fb, ud, yv]

if __name__=="__main__":
    drone_cam = cv2.namedWindow("MogDrone", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MogDrone", (640, 640))
    # tello = Tello()
    # tello.connect()
    # tello.streamon()
    # print("Tello Battery: " + str(tello.get_battery()))
    score = randrange(100)
    pygame.init()
    win = pygame.display.set_mode((400, 400))
    time_s = perf_counter()
    time_e = 0.0

    while True:
        _, frame = cap.read()
        # vals = process_key_in(tello)
        # tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])
        
        # frame = cv2.resize(tello.get_frame_read().frame, (640, 640))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        RGBFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = hog_face_detector(gray)
        for i in range(0, len(faces)):

            face_landmarks = dlib_facelandmark(gray, faces[i])

            for n in range(0, 68):
                
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                if(n == 67):
                    cv2.circle(RGBFrame, (x, y), 1, (0, 0, 0), 1)
                else:
                    cv2.circle(RGBFrame, (x, y), 1, (0, 255, 255), 1)
            time_e = perf_counter()
            if (time_e-time_s) > 1.0:
                score = rizz_calculator(face_landmarks)
                time_s = perf_counter()
            RGBFrame = cv2.putText(RGBFrame, "MogScore of face " + str(i+1) + ": " + str(score), (100*i, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            
        cv2.imshow("MogDrone", RGBFrame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.imwrite("mog_capture.jpg", RGBFrame)
            break

        sleep(0.05)
    cap.release()
    cv2.destroyAllWindows()
    # tello.streamoff()