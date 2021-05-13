import cv2
import glob

videos = glob.glob('./Videos/*.mp4')

for v in videos:
    cap = cv2.VideoCapture(v)
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))