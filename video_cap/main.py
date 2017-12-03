import argparse
from video_cap.videocap import VideoCap

parser = argparse.ArgumentParser('Video capture in python using opencv lib')
parser.add_argument('--fun', type=str,
                    choices=(
                        'threshold', 'gauss-threshold', 'sobel-det', 'laplacian-det', 'candy-edge-det', 'face-rec'),
                    help="Available functions: threshold, gauss-threshold, sobel-det,"
                         " laplacian-det, candy-edge-det, face-rec", required=True)
parser.add_argument('--x', type=int, choices=range(0, 640), help="X resolution 0 to 640", required=True)
parser.add_argument('--y', type=int, choices=range(0, 480), help="Y resolution 0 to 480", required=True)


def main():
    argumets = parser.parse_args()
    video_cap = VideoCap(argumets.fun, argumets.x, argumets.y)
    video_cap.run()
