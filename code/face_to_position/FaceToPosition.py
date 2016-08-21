import multiprocessing
import numpy as np
from matplotlib import pyplot as plt

import sharedmem


class FaceToPosition(multiprocessing.Process):
    def __init__(self, face_width, res_x, res_y, f_x, f_y, fan_position, visualize=False):

        # Initialize multiprocessing.Process parent
        multiprocessing.Process.__init__(self)

        # Exit event for stopping process
        self._exit = multiprocessing.Event()

        # Event that is set, everytime a new servo angle position has been computed
        self.newposition_event = multiprocessing.Event()

        # An array in shared memory for storing the current face position
        self._currentface = sharedmem.empty((4, 1), dtype='int16')

        # An array in shared memory for storing the current position angles
        self._currentangles = sharedmem.empty((2, 1), dtype='float')

        self._facewidth = face_width
        self._res_x = res_x
        self._res_y = res_y
        self._f_x = f_x
        self._f_y = f_y
        self._fan_position = fan_position

        # Defines whether to visualize the servo angles position
        self._visualize = visualize

    def run(self):
        # Clear events
        self._exit.clear()
        self.newposition_event.clear()

        if self._visualize:
            # Interactive plotting mode
            plt.ion()
            f, axarray = plt.subplots(2, 1)
            f.subplots_adjust(hspace=.5)


        # While exit event is not set...
        while not self._exit.is_set():
            # ..clear new face event
            self.newposition_event.clear()

            # ...get face parameters
            x = np.float64(self._currentface[0])
            y = np.float64(self._currentface[1])
            w = np.float64(self._currentface[2])
            h = np.float64(self._currentface[3])

            # ...flip y coordinate
            #y = h - y
            h = -h

            # ...compute position of face center
            face_center = [x + w / 2, y - h / 2]

            # ...compute distances from camera to face in mm
            c1_horizontal = self._facewidth * self._f_x / w
            c1_vertical = self._facewidth * self._f_y / h

            # ...compute face distances from camera center
            a1_horizontal = (face_center[0] - self._res_x / 2) * ((self._facewidth / 2.0) / (w / 2.0))
            a1_vertical = (face_center[1] - self._res_y / 2) * ((self._facewidth / 2.0) / (h / 2.0))

            # ...compute distance from camera to face plane
            b1_horizontal = np.sqrt(c1_horizontal ** 2 - a1_horizontal ** 2)
            b1_vertical = np.sqrt(c1_vertical ** 2 - a1_vertical ** 2)

            # ...compute a2
            a2_horizontal = self._fan_position[0] - a1_horizontal
            a2_vertical = self._fan_position[2] - a1_vertical

            # ...compute b2
            b2_horizontal = b1_horizontal - self._fan_position[1]
            b2_vertical = b1_vertical - self._fan_position[1]

            # ...compute c2
            c2_horizontal = np.sqrt(a2_horizontal ** 2 + b2_horizontal ** 2)
            c2_vertical = np.sqrt(a2_vertical ** 2 + b2_vertical ** 2)

            # ...compute alphas
            alpha_horizontal = np.arccos(
                (b2_horizontal ** 2 + c2_horizontal ** 2 - a2_horizontal ** 2) / (2 * b2_horizontal * c2_horizontal))
            if a1_horizontal > self._fan_position[0]:
                alpha_horizontal = alpha_horizontal * -1

            alpha_vertical = np.arccos(
                (b2_vertical ** 2 + c2_vertical ** 2 - a2_vertical ** 2) / (2 * b2_vertical * c2_vertical))
            if a1_vertical > self._fan_position[2]:
                alpha_vertical = alpha_vertical * -1

            # TESTING: angle offsets
            offset_horizontal = 0
            offset_vertical = 0

            # Convert angles from radian to degree and copy them into shared memory
            self._currentangles[0] = (alpha_horizontal * (180 / np.pi) - offset_horizontal).copy()
            self._currentangles[1] = (alpha_vertical * (180 / np.pi) - offset_vertical).copy()

            # Set event
            self.newposition_event.set()

            if self._visualize:
                # Visualize stuff
                axarray[0].cla()
                axarray[0].plot(0, 0, 'ok')
                axarray[0].plot(a1_horizontal, b1_horizontal, 'or')
                axarray[0].plot(self._fan_position[0], self._fan_position[1], 'og')
                x_fan = self._fan_position[0] + (c2_horizontal * np.cos(np.pi / 2 + alpha_horizontal))
                y_fan = self._fan_position[1] + (c2_horizontal * np.sin(np.pi / 2 + alpha_horizontal))
                axarray[0].plot([x_fan, self._fan_position[0]], [y_fan, self._fan_position[1]], '--y')
                axarray[0].set_xlim([1000, -1000])
                axarray[0].set_ylim([1500, -300])
                axarray[0].set_title('top view')
                axarray[0].set_xlabel('x [mm]')
                axarray[0].set_ylabel('y [mm]')
                axarray[0].set_aspect('equal', adjustable='box')
                axarray[0].text(0, -100,
                                'cam',
                                horizontalalignment='center', verticalalignment='center')
                axarray[0].text(self._fan_position[0], self._fan_position[1] - 100,
                                'fan',
                                horizontalalignment='center', verticalalignment='center')
                axarray[0].text(0, -100,
                                'cam',
                                horizontalalignment='center', verticalalignment='center')
                if not np.isinf(a1_horizontal):
                    axarray[0].text(a1_horizontal, b1_horizontal + 100,
                                    'face',
                                    horizontalalignment='center', verticalalignment='center')

                axarray[1].cla()
                axarray[1].plot(0, 0, 'ok')
                axarray[1].plot(b1_vertical, a1_vertical, 'or')
                axarray[1].plot(self._fan_position[1], self._fan_position[2], 'og')
                z_fan = self._fan_position[2] + (c2_vertical * np.cos(np.pi / 2 + alpha_vertical))
                y_fan = self._fan_position[1] + (c2_vertical * np.sin(np.pi / 2 + alpha_vertical))
                axarray[1].plot([y_fan, self._fan_position[1]], [z_fan, self._fan_position[2]], '--y')
                axarray[1].set_xlim([-300, 1200])
                axarray[1].set_ylim([-500, 500])
                axarray[1].set_title('side view')
                axarray[1].set_xlabel('y [mm]')
                axarray[1].set_ylabel('z [mm]')
                axarray[1].set_aspect('equal', adjustable='box')
                axarray[1].text(0, 60,
                                'cam',
                                horizontalalignment='center', verticalalignment='center')
                axarray[1].text(self._fan_position[1], self._fan_position[2] - 60,
                                'fan',
                                horizontalalignment='center', verticalalignment='center')
                if not np.isinf(a1_horizontal):
                    axarray[1].text(b1_vertical, a1_vertical + 60,
                                    'face',
                                    horizontalalignment='center', verticalalignment='center')
                plt.pause(0.001)

    def terminate(self):
        # Set exit event
        self._exit.set()

    def set_face(self, face):
        self._currentface[:] = face.copy()

    def get_angles(self):
        return self._currentangles
