import cv2
import os
import threading


def jsonToNumpy():
    pass

def extractJson(path):
    if not os.path.isdir('json'):
        os.mkdir('json')

    print()
    print('Exracting json... from ' + path)
    os.system('cd openpose && ./build/examples/openpose/openpose.bin --video ' + path + ' --write_json ../json/ --display 0 --render_pose 0 --model_pose COCO')

    return path.split('/')[-1][:-4], 'ok'

class Webcam(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.frame_size = (640, 480)
        self.target_frames = 32
        self.cnt = 0
        self.num = 0
        self.v_path = './videos/video_' + str(self.num) + '.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 압축 알고리즘
        self.out = cv2.VideoWriter(self.v_path, self.fourcc, 30.0, self.frame_size)

    def saveVideos(self):
        if not os.path.isdir('videos'):
            os.mkdir('videos')

        while True:
            ret, frame = self.cap.read()  # 1 프레임씩 캡처
            # print(cnt)

            # frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1) # horizontal flip
            frame = cv2.resize(frame, self.frame_size)

            if not ret:
                break

            self.num = self.cnt // 32
            # print(self.num)

            if self.num >= 1 and self.cnt % 32 == 0:
                # print(self.v_path)
                if os.path.isfile(self.v_path):
                    self.out.release()
                    extractJson('../' + self.v_path[2:])
                else:
                    print()
                    print("don't have a video file : " + self.v_path)
                    break
                self.v_path = './videos/video_' + str(self.num) + '.mp4'

                self.out = cv2.VideoWriter(self.v_path, self.fourcc, 30.0, self.frame_size)
                # print(self.v_path)

            self.out.write(frame)

            self.cnt += 1

            cv2.imshow('frame', frame)

            key = cv2.waitKey(25)
            if key == 27 and self.cnt % 32 == 0:
                break  # ESC

        self.cap.release()
        self.out.release()

webcam = Webcam()
webcam.saveVideos()