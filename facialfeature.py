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

def sig(x):
    return 1/(1 + np.exp(-x))

def calc_sym(f):
    coors = []
    for i in range(0, 68):
        coors.append(np.array([[f.part(i).x], 
                               [f.part(i).y]]))
    points = np.array(coors)
    x_mean = np.mean(points[:, 0])
    y_mean = np.mean(points[:, 1])
    
    # Reflect points across the vertical line (y-axis symmetry)
    reflected_x = 2 * x_mean - points[:, 0]
    distances_x = np.sqrt((reflected_x - points[:, 0])**2 + (points[:, 1] - points[:, 1])**2)
    symmetry_score_x = np.mean(distances_x)
    
    # Reflect points across the horizontal line (x-axis symmetry)
    reflected_y = 2 * y_mean - points[:, 1]
    distances_y = np.sqrt((points[:, 0] - points[:, 0])**2 + (reflected_y - points[:, 1])**2)
    symmetry_score_y = np.mean(distances_y)
    return sig((symmetry_score_x+symmetry_score_y)/320)

def calc_mouth(f):
    right_mouth = f.part(54).x
    left_mouth = f.part(48).x

    right_eye = (f.part(42).x + f.part(44).x)/2
    left_eye = (f.part(36).x + f.part(39).x)/2

    total_diff = (right_mouth+right_eye+left_eye+left_mouth)/4 - 320
    print(total_diff/10)
    return sig(total_diff/10)

def calc_eyes(f):
    o = f.part(27).x - f.part(8).x
    a = f.part(27).y - f.part(8).y
    angle = 0
    if a != 0:
        angle = atan2(o, a)

    eye_points = [f.part(36) , f.part(39), f.part(42), f.part(45)]
    vecs = []
    for i in range(len(eye_points)):
        vecs.append(np.array([[eye_points[i].x],
                             [eye_points[i].y]]))
    rot = np.array([[cos(angle), sin(angle)],
                   [-sin(angle), cos(angle)]])
    
    for i in range(len(vecs)):
        vecs[i] = np.matmul(rot, vecs[i])
    
    score = ((vecs[1])[1,0]-(vecs[0])[1,0]) + ((vecs[2])[1,0]-(vecs[3])[1,0])
    return sig(score)

def rizz_calculator(face):
    weights = {"mouth": 50, "eyes": 20, "sym": 30}
    return weights["mouth"]*calc_mouth(face) + weights["eyes"]*calc_eyes(face) + weights["sym"]*calc_sym(face)

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
        update_score = False
        for i in range(0, len(faces)):
        
            face_landmarks = dlib_facelandmark(gray, faces[i])

            for n in range(0, 68):
                
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                cv2.circle(RGBFrame, (x, y), 1, (0, 255, 255), 1)
            time_e = perf_counter()
            if (time_e-time_s) > 1.0:
                score = rizz_calculator(face_landmarks)
                update_score = True
            RGBFrame = cv2.putText(RGBFrame, "MogScore of face " + str(i+1) + ": " + str(score), (0, 100*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if update_score:
            time_s = perf_counter()

        cv2.imshow("MogDrone", RGBFrame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.imwrite("mog_capture.jpg", RGBFrame)
            break
        sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()
    # tello.streamoff()