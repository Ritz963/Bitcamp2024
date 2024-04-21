import keyboard
import time

while True:  # making a loop
    try:  # used try so that if user pressed other than the given key error will not be shown
        if keyboard.is_pressed('w'):
            print("forward")
            time.sleep(.25)
        elif keyboard.is_pressed('s'):
            print("backward")
            time.sleep(.25)
        elif keyboard.is_pressed('a'):
            print("right")
            time.sleep(.25)
        elif keyboard.is_pressed('d'):
            print("left")
            time.sleep(.25)
        elif keyboard.is_pressed('q'):
            print('up')
            time.sleep(.25)
        elif keyboard.is_pressed('e'):
            print('down')
            time.sleep(.25)
        else:
            pass
    except:
        break
    