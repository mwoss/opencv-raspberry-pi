import numpy as np
import cv2

_PATH = 'Trained_Models/haarcascade_frontalface_default.xml'


class VideoCap:
    def __init__(self, function_name, x, y):
        self.cap = cv2.VideoCapture(0)
        self.face_haar_cascade = None if function_name == 'face-det' else cv2.CascadeClassifier(_PATH)
        self.fun = getattr(VideoCap, self.functions[function_name][0])
        self.x_bound = x
        self.y_bound = y

    def run(self):
        while True:
            _, frame = self.cap.read()
            # function here
            # mozliwosc uzycia kamerki bez funckji, wszystke opecje opcjonalne
            edge = self.fun(self, frame, 150, 170)
            cv2.imshow('frame', edge)
            if cv2.waitKey(1) % 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def get_arguemnts(self, args_list):
        pass

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
        return cv2.Sobel(gray_scale, cv2.CV_64F, x, y, ksize=5)

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
            # roi_gray = gray[y:y + h, x:x + w]
            # roi_color = frame[y:y + h, x:x + w]
        return frame

    functions = {
        'face-rec': ['_face_detection'],
        'threshold': ['_threshhold_filter', 'threshold'],
        'gauss-threshold': ['_threshold_gauss', 'block_size', 'c'],
        'sobel-det': ['_sobel_edge_det', 'x', 'y'],
        'laplacian-det': ['_laplacian_edge_detecion'],
        'candy-edge-det': ['_canny_edge_det', 'min', 'max'],
    }





    # cut_frame = frame[0:240, 0:640]

    ##      threshold = threshhold_filter(frame, 120)
    # threshold = threshold_gauss(frame, 115, 1)
    # frame[200:400, 400:600] = threshold
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', gray)



    # edge = canny_edge_det(frame, 150, 170)
    # frame[0:240, 0:640] = edge
