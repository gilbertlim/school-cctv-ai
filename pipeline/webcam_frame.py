import cv2
import os
import glob
import threading
from queue import Queue
import numpy as np
from preprocessing_frame import Preprocessing
from tensorflow import keras

np.set_printoptions(precision=4, suppress=True)

q_nparray = Queue()
q_predict = Queue()
model = keras.models.load_model('./models/lstm_p1_punching_smoking_walking_running_angle.h5')

classes = {0 : 'walking', 1 : 'smoking', 2: 'punching', 3 : 'running'}

def predict():
    print('q_nparray_length : ', len(q_nparray.queue))
    # print('q_nparray : ', q_nparray.queue)

    inputs = []
    nfpeople = 0

    if len(q_nparray.queue) >= 32:
        for i in range(32):
            inputs.append(q_nparray.get())
        nfpeople = len(inputs[0])
    # print('inputs', inputs)

    # 사람 수에 따라 변수 선언(동적 변수)
    for i in range(nfpeople):
        globals()['p_' + str(i)] = []

    # 사람별 데이터 분리(p_1, p_2, p_N, ...)
    for ip in inputs:
        for i in range(len(ip)):
            globals()['p_' +str(i)].append(ip[i])

    predicted = []
    for i in range(nfpeople):
        globals()['p_' +str(i)] = np.array(globals()['p_' +str(i)])
        globals()['p_' +str(i)] = globals()['p_' +str(i)].reshape(-1, 32, 8)

        output = model.predict(globals()['p_' + str(i)])
        output = np.argmax(output[0], axis=-1)
        output = classes[int(output)]
        predicted.append(output)

    q_predict.put(predicted)

    print('predicted', predicted)

def jsonToNumpy():
    j_list = sorted(glob.glob('./json/*.json'))

    if len(j_list) >= 2:
        print('\nModify json to numpy... ', j_list[0] + '\n---------------------------------------------')
        prep = Preprocessing(j_list[0])
        angle = prep.main()
        q_nparray.put(angle)

        os.remove(j_list[0])
    else:
        print("\ndoesn't exist json files\n---------------------------------------------")

def extractJson(path):
    if not os.path.isdir('json'):
        os.mkdir('json')

    print('\n---------------------------------------------\nExracting json... from ' + path)
    os.system('cd openpose && ./build/examples/openpose/openpose.bin --video ' + path + ' --write_json ../json/ --display 0 --render_pose 0 --model_pose COCO')

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
        if not os.path.isdir('./videos'):
            os.mkdir('./videos')

        while True:
            ret, frame = self.cap.read()  # 1 프레임씩 캡처

            if not ret:
                print('ret == False')
                break

            frame = cv2.flip(frame, 1)  # horizontal flip
            frame = cv2.resize(frame, self.frame_size)

            cv2.imshow('frame', frame)

            self.num = self.cnt // self.target_frames

            if self.num >= 1 and self.cnt % self.target_frames == 0:
                if os.path.isfile(self.v_path):
                    print('\nSave video... ' + self.v_path)

                    self.out.release()
                    t = threading.Thread(target=extractJson, args=('../' + self.v_path[2:],))
                    # t.daemon = True
                    t.start()

                    t2 = threading.Thread(target=jsonToNumpy)
                    # t2.daemon = True
                    t2.start()

                    t3 = threading.Thread(target=predict)
                    # t3.daemon = True
                    t3.start()

                else:
                    print("\ndon't have a video file : " + self.v_path)
                    break

                self.v_path = './videos/video_' + str(self.num) + '.mp4'
                self.out = cv2.VideoWriter(self.v_path, self.fourcc, 30.0, self.frame_size)
                # print(self.v_path)

            # self.out.write(frame)

            self.cnt += 1

            # ESC 누르면 화면 캡쳐 종료
            key = cv2.waitKey(25)
            if key == 27 and self.cnt % self.target_frames == 0:
                break

        self.cap.release()
        self.out.release()

webcam = Webcam()
webcam.saveVideos()