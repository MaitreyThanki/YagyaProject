from machine import Pin, PWM
from time import sleep

# Pin setup
DIR = Pin(2, Pin.OUT)
pwm = PWM(Pin(3))
pwm.freq(1000)   # 1 kHz PWM

def motor_forward(speed=60000):
    DIR.value(1)  # direction HIGH
    pwm.duty_u16(speed)
    print("Forward")

def motor_backward(speed=60000):
    DIR.value(0)  # direction LOW
    pwm.duty_u16(speed)
    print("Backward")

def motor_stop():
    pwm.duty_u16(0)
    print("Stopped")

# Test loop
while True:
    motor_forward()
    sleep(2)

    motor_stop()
    sleep(1)

    motor_backward()
    sleep(2)

    motor_stop()
    sleep(1)

