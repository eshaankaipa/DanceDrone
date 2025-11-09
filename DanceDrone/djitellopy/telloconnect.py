from djitellopy import Tello
import time

tello = Tello()
    
tello.connect()

print(tello.get_battery())
    
