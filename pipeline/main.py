from tracker import Target
from webcam import Webcam

path = './Videos'
tracker = Target(path)

webcam = Webcam()


if __name__ == '__main__':
    tracker.start()
    webcam.saveVideos().start()