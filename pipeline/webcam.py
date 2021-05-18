import cv2
import os
import threading

class Webcam(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self. frame_size = (640, 480)
        self.target_frames = 32
        self.cnt = 0
        self.num = 0
        self.v_path = './Videos/video_' + str(self.num) + '.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 압축 알고리즘
        self.out = cv2.VideoWriter(self.v_path, self.fourcc, 30.0, self.frame_size)

    def saveVideos(self):
        if not os.path.isdir('./Videos'):
            os.mkdir('Videos')

        while True:
            ret, frame = self.cap.read()  # 1 프레임씩 캡처
            # print(cnt)

            # frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1) # horizontal flip
            frame = cv2.resize(frame, self.frame_size)

            if not ret:
                break

            num = self.cnt // 32
            # print(num)

            if self.cnt % 32 == 0:
                v_path = './Videos/video_' + str(num) + '.mp4'
                self.out = cv2.VideoWriter(v_path, self.fourcc, 30.0, self.frame_size)
                # print(v_path)

            self.out.write(frame)

            self.cnt += 1

            cv2.imshow('frame', frame)

            key = cv2.waitKey(25)
            if key == 27 and self.cnt % 32 == 0:
                break  # ESC

        self.cap.release()
        self.out.release()

# webcam = Webcam()
# webcam.saveVideos()