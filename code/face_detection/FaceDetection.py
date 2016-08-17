import multiprocessing
import cv2
import sharedmem
from Filter import lowpass


class FaceDetection(multiprocessing.Process):
    def __init__(self, x, y, scale_factor=1.1, minsize=(60, 60),
                 classifier='haarcascade_frontalface_alt2.xml',
                 use_lowpass=True, lowpass_rc=50,
                 visualize=False):

        # Initialize multiprocessing.Process parent
        multiprocessing.Process.__init__(self)

        # Exit event for stopping process
        self._exit = multiprocessing.Event()

        # Event that is set, everytime a face is detected
        self.newface_event = multiprocessing.Event()

        # Event that pauses the main loop if set
        self._pause_event = multiprocessing.Event()

        # An array in shared memory to store the current image frame
        self._currentframe = sharedmem.empty((y, x), dtype='uint8')

        # Set camera parameters
        self._x = x
        self._y = y

        # Set parameters for face detection algorithm
        self._scale_factor = scale_factor
        self._minsize = minsize
        self._classifier_file = classifier
        self._use_lowpass = use_lowpass
        self._lowpass_rc = lowpass_rc

        # Defines whether to visualize the camera output
        self._visualize = visualize

        # A tuple for storing the current width and height of a face
        self._currentface = sharedmem.empty((4, 1), dtype='float')

        # A tuple for storing the last width and height of a face
        self._lastface = sharedmem.empty((4, 1), dtype='float')

        # Setup a multiscale classifier
        self._classifier = cv2.CascadeClassifier(self._classifier_file)

    def run(self):
        # Clear events
        self._exit.clear()
        self.newface_event.clear()

        # While exit event is not set...
        while not self._exit.is_set() or not self._pause_event.is_set():
            # ..clear new face event
            self.newface_event.clear()

            # ...try to detect face
            face = self._detect_face(self._currentframe, self._scale_factor, self._minsize)
            # ...if face is detected ...
            if face != ():
                # ...get boundaries of face in pixels
                for (x, y, w, h) in face:

                    if self._use_lowpass:
                        # ...apply temporal lowpass filter to face boundary
                        self._currentface[0] = lowpass(x, self._lastface[0], self._lowpass_rc, 33.333).copy()

                        self._currentface[1] = lowpass(y, self._lastface[1], self._lowpass_rc, 33.333).copy()

                        self._currentface[2] = lowpass(w, self._lastface[2], self._lowpass_rc, 33.333).copy()

                        self._currentface[3] = lowpass(h, self._lastface[3], self._lowpass_rc, 33.333).copy()

                    else:
                        # ...copy coordinates into shared memory
                        self._currentface[0] = x.copy()
                        self._currentface[1] = y.copy()
                        self._currentface[2] = w.copy()
                        self._currentface[3] = h.copy()

                if self._visualize:
                    # ...draw a rectangle around the boundaries
                    cv2.rectangle(self._currentframe,
                                  (int(self._currentface[0]), int(self._currentface[1])),
                                  (int(self._currentface[0]) + int(self._currentface[2]),
                                   int(self._currentface[1]) + int(self._currentface[3])),
                                  (0, 255, 0), 2)

                    # ...display image
                    cv2.imshow('FaceDetection', self._currentframe)
                    cv2.waitKey(1)

                self._lastface = self._currentface.copy()
                self.newface_event.set()

        # If exit event set...
        if self._exit_event.is_set():
            # ...close windows
            cv2.destroyAllWindows()

    def terminate(self):
        # Set exit event
        self._exit.set()

    def _detect_face(self, frame, scalefactor=1.1, minsize=(60, 60), flags=(cv2.cv.CV_HAAR_SCALE_IMAGE +
                                                                            cv2.cv.CV_HAAR_DO_CANNY_PRUNING +
                                                                            cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT +
                                                                            cv2.cv.CV_HAAR_DO_ROUGH_SEARCH)):
        # Detect face using a multiscale haar classifier
        face = self._classifier.detectMultiScale(frame,
                                                 scaleFactor=scalefactor,
                                                 minSize=minsize,
                                                 flags=flags)

        return face

    def set_frame(self, image):
        self._currentframe[:] = image.copy()

    def get_face(self):
        return self._currentface


if __name__ == "__main__":
    import sys

    sys.path.insert(0, '../unwarping')

    from Unwarping import Unwarping
    import numpy as np

    # --- Parameters ---
    x = 640
    y = 480
    f_x = 673.9683892
    f_y = 676.08466459
    c_x = 343.68638231
    c_y = 245.31865398
    # distortion coefficients
    d = 5.44787247e-02, 1.23043244e-01, -4.52559581e-04, 5.47011732e-03, -6.83110234e-01

    # Construct camera matrix
    K = np.array([[f_x, 0., c_x],
                  [0., f_y, c_y],
                  [0., 0., 1.]])

    # --- Tasks ---
    # Start unwarping task
    unwarper = Unwarping(x, y, K, d, visualize=True, debug=False)
    unwarper.start()

    # Start face detection task
    face_detection = FaceDetection(x, y, visualize=True)
    face_detection.start()

    # --- Main ---
    while True:
        # Wait for new image frame from unwarper
        unwarper.newframe_event.wait()
        # Get frame from unwarper
        unwarped_frame = unwarper.get_current_frame()

        # Pass frame into face detection task
        face_detection.set_frame(unwarped_frame)
        # Wait for new face to be detected
        face_detection.newface_event.wait(0.0333)
