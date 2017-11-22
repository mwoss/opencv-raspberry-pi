import numpy as np
import cv2


def threshhold_filter(frame, threshold, max_value=255, \
                      threshold_type=cv2.THRESH_BINARY):
    ret, treshold_frame = cv2.threshold(frame, threshold, max_value, threshold_type)
    return treshold_frame


def threshold_gauss(frame, block_size, c, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                    threshold_type=cv2.THRESH_BINARY):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaus = cv2.adaptiveThreshold(gray, max_value, adaptive_method, threshold_type, block_size, c)
    matrix_fit = np.reshape(gaus, gaus.shape + (1,))
    return matrix_fit


def sobel_edge_det(frame, x, y):
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Sobel(gray_scale, cv2.CV_64F, x, y, ksize=5)


def laplacian_edge_detecion(frame):
    return cv2.Laplacian(frame, cv2.CV_64F)


def canny_edge_det(frame, min, max):
    canny = cv2.Canny(frame, min, max)
    return np.reshape(canny, canny.shape + (1,))


def nothing(x):
    pass


if __name__ == '__main__':
    # inicjalizacja kamery argument to numer podlaczonej kamerki: pierwsza=0,druga=1 itd.
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        cut_frame = frame[0:240, 0:640]

        ##      threshold = threshhold_filter(frame, 120)
        # threshold = threshold_gauss(frame, 115, 1)
        # frame[200:400, 400:600] = threshold
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('frame', gray)



        edge = canny_edge_det(frame, 150, 170)
        # frame[0:240, 0:640] = edge

        cv2.imshow('frame', edge)
        if cv2.waitKey(1) % 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
