import math
import multiprocessing
import time
import copy
import sharedmem

from Filter import lowpass


class ServoControl(multiprocessing.Process):
    def __init__(self, servo_horizontal, m_horizontal, b_horizontal,
                 servo_vertical, m_vertical, b_vertical,
                 f, speed, pwm_min, pwm_max):

        # Initialize multiprocessing.Process parent
        multiprocessing.Process.__init__(self)

        # Exit event for stopping process
        self._exit = multiprocessing.Event()

        # Set servo pins
        self._s_h = servo_horizontal
        self._s_v = servo_vertical

        # Set slopes
        self._m_horizontal = m_horizontal
        self._m_vertical = m_vertical

        # Set intercepts
        self._b_horizontal = b_horizontal
        self._b_vertical = b_vertical

        # Set update frequency
        self._f = 1.0 / f

        # Set increment per time step
        self._increment = speed / f

        # Set minimum and maximum pwm ranges
        self._pwm_min = pwm_min
        self._pwm_max = pwm_max

        # An array in shared memory for storing the new desired servo position angles
        self._newangles = sharedmem.zeros((2, 1), dtype='float')

        # A list containing the current servo angles
        self._currentangles = sharedmem.ones((2, 1), dtype='float')

        # Initialize ServoBlaster device
        self._servoblaster = open('/dev/servoblaster', 'w')

    def __del__(self):
        self._servoblaster.close()

    def run(self):
        # Clear events
        self._exit.clear()

        self._set_servo_pwm(self._s_h, self._currentangles[0])
        self._set_servo_pwm(self._s_v, self._currentangles[1])

        old_horizontal_angle = self._currentangles[0]
        old_vertical_angle = self._currentangles[1]

        # While exit event is not set...
        while not self._exit.is_set():

            # ...increment or decrement horizontal servo angle depending on speed and update frequency
            if self._newangles[0] > self._currentangles[0]:
                self._currentangles[0] += self._increment
            elif self._newangles[0] < self._currentangles[0]:
                self._currentangles[0] -= self._increment

            # ...increment or decrement vertical servo angle depending on speed and update frequency
            if self._newangles[1] > self._currentangles[1]:
                self._currentangles[1] += self._increment
            elif self._newangles[1] < self._currentangles[1]:
                self._currentangles[1] -= self._increment

            # ...apply temporal lowpass filter to angles
            new_horizontal_angle = lowpass(self._currentangles[0],
                                           old_horizontal_angle,
                                           0.1, self._f)

            new_vertical_angle = lowpass(self._currentangles[0],
                                         old_vertical_angle,
                                         0.1, self._f)

            # ...store angles
            old_horizontal_angle = new_horizontal_angle
            old_vertical_angle = new_vertical_angle

            # ...set servo pwm
            self._set_servo_pwm(self._s_h, self.angle_to_pwm(new_horizontal_angle,
                                                             self._m_horizontal, self._b_horizontal))
            self._set_servo_pwm(self._s_v, self.angle_to_pwm(new_vertical_angle,
                                                             self._m_vertical, self._b_vertical))

            time.sleep(self._f)

    def set_new_angles(self, horizontal_angle, vertical_angle):
        self._newangles[0] = copy.copy(horizontal_angle)
        self._newangles[1] = copy.copy(vertical_angle)

    def _set_servo_pwm(self, servo, pwm):
        if pwm in range(self._pwm_min, self._pwm_max, 1):
            self._servoblaster.write(str(servo) + '=' + str(pwm) + '\n')
            self._servoblaster.flush()

    def angle_to_pwm(self, angle, m, b):
        if math.isnan(angle):
            return self._pwm_min
        else:
            pwm = m * angle + b
            return int(pwm)


if __name__ == '__main__':
    servocontrol = ServoControl(1, 1.2290352706, 166.3342587025,
                                3, -1.2517764093, 166.7882520133,
                                100, 100, 90, 226)
    servocontrol.start()

    while True:
        servocontrol.set_new_angles(30, 30)
        time.sleep(1)
        servocontrol.set_new_angles(-30, -30)
        time.sleep(1)
