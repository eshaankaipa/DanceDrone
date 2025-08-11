from djitellopy import Tello


tello = Tello()

tello.connect()

tello.takeoff()
tello.move_forward(95)
tello.rotate_clockwise(5)
tello.move_forward(130)
tello.rotate_counter_clockwise(100)
tello.move_forward(130)
tello.move_up(50)
tello.rotate_counter_clockwise(90)
tello.move_forward(100)

tello.land()
