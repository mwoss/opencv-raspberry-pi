import argparse
from video_cap.videocap import VideoCap

parser = argparse.ArgumentParser('Video capture in python using opencv lib')
parser.add_argument('--fun', type=str,
                    choices=(
                        'threshold', 'gauss-threshold', 'sobel-det', 'laplacian-det', 'canny-edge-det', 'face-rec',
                        'none'),
                    help="Available functions: threshold, gauss-threshold, sobel-det,"
                         " laplacian-det, canny-edge-det, face-rec, none", required=True, metavar="FUNC_NAME")
parser.add_argument('-x', type=int, nargs=2, default=[0, 0],
                    help="Bound of filter function on X axis. First val < Second val",
                    metavar='X_RESOLUTIONS'
                    )
parser.add_argument('-y', type=int, nargs=2, default=[0, 0],
                    help="Bound of filter function on Y axis. First val < Second val",
                    metavar='Y_RESOLUTIONS')


def main():
    arguments = parser.parse_args()
    if arguments.x[0] > arguments.x[1] or arguments.y[0] > arguments.y[1]:
        raise AttributeError("First val < Second val!")

    video_cap = VideoCap(arguments.fun, arguments.x[0], arguments.y[0], arguments.x[1], arguments.y[1])
    video_cap.run()
