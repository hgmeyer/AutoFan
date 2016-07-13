import multiprocessing
import time

import cv2
import sharedmem


class Unwarping(multiprocessing.Process):
    def __init__(self, x, y, K, d, camid=0, visualize=False, debug=False):

        # Initialize multiprocessing.Process parent
        multiprocessing.Process.__init__(self)

        # Exit event for stopping process
        self._exit = multiprocessing.Event()

        # Event that is set, everytime an image has been unwarped
        self.newframe_event = multiprocessing.Event()

        # Event that pauses the main loop if set
        self._pause_event = multiprocessing.Event()

        # Defines whether to visualize the camera output
        self._visualize = visualize

        # Switches debugging mode
        self._debug = debug

        # Some variable for storing the time of the last frame
        self._oldtime = time.time()

        # Set camera parameters
        self._cam_device_id = camid  # Get camera ID
        self._x = x  # Get width
        self._y = y  # Get height

        # An empty array in shared memory to store the current image frame
        self._currentframe = sharedmem.empty((y, x), dtype='uint8')

        # Define camera matrix K
        self._K = K

        # Define distortion coefficients d
        self._d = d

        # Setup camera object using OpenCV
        self._cam = cv2.VideoCapture(self._cam_device_id)
        self._cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, self._x)
        self._cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, self._y)

        # Generate optimal camera matrix
        self._newcameramatrix, self._roi = cv2.getOptimalNewCameraMatrix(self._K, self._d, (self._x, self._y), 0)

        # Generate LUTs for undistortion
        self._mapx, self._mapy = cv2.initUndistortRectifyMap(self._K, self._d, None, self._newcameramatrix,
                                                             (self._x, self._y), 5)

    def run(self):
        # Clear events
        self._exit.clear()

        # While exit event is not set...
        while not self._exit.is_set() or not self._pause_event.is_set():
            # ...clear new frame event
            self.newframe_event.clear()

            # ... read a frame from the camera
            frame = self._cam.read()[1]

            # ...convert camera image to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ...remap image into shared memory
            self._currentframe[:] = cv2.remap(frame, self._mapx, self._mapy, cv2.INTER_LINEAR).copy()

            if self._visualize:
                cv2.imshow('Unwarped image', self._currentframe)
                cv2.waitKey(1)

            if self._debug:
                print str(1 / (time.time() - self._oldtime)) + " frames/sec"
                self._oldtime = time.time()

            # Set new frame event
            self.newframe_event.set()

        # If exit event set...
        if self._exit_event.is_set():
            # ...release camera
            self._cam.release()
            # ...close windows
            cv2.destroyAllWindows()

    def terminate(self):
        # Set exit event
        self._exit.set()

    def get_current_frame(self):
        return self._currentframe
