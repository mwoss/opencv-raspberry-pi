import numpy as np
import cv2

_PATH = 'video_cap/resources/trained_models/haarcascade_frontalface_default.xml'


class VideoCap:
    def __init__(self, function_name, x_min, y_min, x_max, y_max):
        self.cap = cv2.VideoCapture(0)
        self.face_haar_cascade = None if function_name != 'face-rec' else cv2.CascadeClassifier(_PATH)
        self.func_name = function_name
        self.fun = getattr(VideoCap, self.functions[function_name][0])
        self.args = []
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def run(self):
        self.get_arguments()
        while True:
            _, frame = self.cap.read()
            try:
                cut_frame = frame[self.x_min:self.x_max, self.y_min:self.y_max]
                f_frame = self.fun(self, cut_frame, *self.args)
                frame[self.x_min:self.x_max, self.y_min:self.y_max] = f_frame
            except cv2.error:
                print("Improperly defined resolution bounds or parameter values")
                break
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) % 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def get_arguments(self):
        print("Input user defined function arguments :)")
        for arg in self.functions[self.func_name][1:]:
            self.args.append(int(input("Please, input " + str(arg) + " parameter value: ")))

    def _none_filter(self, frame):
        return frame

    def _threshold_gauss(self,
                         frame,
                         block_size,
                         c,
                         max_value=255,
                         adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                         threshold_type=cv2.THRESH_BINARY):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gauss = cv2.adaptiveThreshold(gray, max_value, adaptive_method, threshold_type, block_size, c)
        matrix_fit = np.reshape(gauss, gauss.shape + (1,))
        return matrix_fit

    def _threshhold_filter(self,
                           frame,
                           threshold,
                           max_value=255,
                           threshold_type=cv2.THRESH_BINARY):

        _, threshold_frame = cv2.threshold(frame, threshold, max_value, threshold_type)
        return threshold_frame

    def _sobel_edge_det(self, frame, x, y):

        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray_scale, cv2.CV_64F, x, y, ksize=5)
        matrix_fit = np.reshape(sobel, sobel.shape + (1,))
        return matrix_fit

    def _laplacian_edge_detecion(self, frame):

        return cv2.Laplacian(frame, cv2.CV_64F)

    def _canny_edge_det(self, frame, min, max):

        canny = cv2.Canny(frame, min, max)
        return np.reshape(canny, canny.shape + (1,))

    def _face_detection(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_haar_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame

    functions = {
        'face-recognition': ['_face_detection'],
        'basic-threshold': ['_threshhold_filter', 'threshold'],
        'gauss-threshold': ['_threshold_gauss', 'block_size', 'c'],
        'sobel-det': ['_sobel_edge_det', 'x derivative degree', 'y derivative degree'],
        'laplacian-det': ['_laplacian_edge_detecion'],
        'canny-edge-det': ['_canny_edge_det', 'min', 'max'],
        'none': ['_none_filter']
    }
