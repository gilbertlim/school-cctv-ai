import cv2
import os
import threading

from glob import glob
from queue import Queue

import numpy as np
import pandas as pd


BODY_PARTS_COCO = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                   [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]

protoFile_coco = '/Users/gilbert/Developer/Project/3_Convergence/School_CCTV_AI/pipeline/openpose/models/pose/coco/pose_deploy_linevec.prototxt'
weightsFile_coco = '/Users/gilbert/Developer/Project/3_Convergence/School_CCTV_AI/pipeline/openpose/models/pose/coco/pose_iter_440000.caffemodel'

queue = Queue()


def output_keypoints(frame, proto_file, weights_file, threshold, model_name, BODY_PARTS, q):
    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # 입력 이미지의 사이즈 정의
    image_height = 368
    image_width = 368

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward()
    # The output is a 4D matrix :
    # The first dimension being the image ID ( in case you pass more than one image to the network ).
    # The second dimension indicates the index of a keypoint.
    # The model produces Confidence Maps and Part Affinity maps which are all concatenated.
    # For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps. Similarly, for MPI, it produces 44 points.
    # We will be using only the first few points which correspond to Keypoints.
    # The third dimension is the height of the output map.
    out_height = out.shape[2]
    # The fourth dimension is the width of the output map.
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    points = []

    print(f"\n============================== {model_name} Model ==============================")
    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        # min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        mapSmooth = cv2.GaussianBlur(prob_map, (3, 3), 0, 0)

        mapMask = np.uint8(mapSmooth >= threshold)

        # find the blobs
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # for each blob find the maxima
        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask

            min_val, prob, min_loc, point = cv2.minMaxLoc(maskedProbMap)

            x = (frame_width * point[0]) / out_width
            x = int(x)
            y = (frame_height * point[1]) / out_height
            y = int(y)

            points.append((x, y))
            # print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    q.put(points)


def extractJson(path):

    print('Exracting json... from ' + path)

    keypoints = []

    cap_32 = cv2.VideoCapture(path)

    while True:
        ret, frame = cap_32.read()

        if not ret:
            print('Done ! ' + path)
            break

        t2 = threading.Thread(target=output_keypoints, args=(frame, protoFile_coco, weightsFile_coco, 0.1, 'COCO', BODY_PARTS_COCO, queue,))
        t2.daemon = True
        t2.start()
        t2.join()

        if len(queue.queue) % 32:
            break

    print(queue.queue)
    cap_32.release()

class MangeQueue:
    def __init__(self):
        self.df = pd.DataFrame(columns=[str(i) for i in range(38)])

class Webcam:
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

            self.num = self.cnt // self.target_frames
            # print(self.num)

            if self.num >= 1 and self.cnt % self.target_frames == 0:
                # print(self.v_path)

                if os.path.isfile(self.v_path):
                    # print()
                    # print('Save video... ' + self.v_path)

                    self.out.release()
                    t = threading.Thread(target=extractJson, args=(self.v_path,))
                    t.daemon = True
                    t.start()

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

            # ESC 누르면 화면 캡쳐 종료
            key = cv2.waitKey(25)
            if key == 27 and self.cnt % self.target_frames == 0:
                break

        self.cap.release()
        self.out.release()


webcam = Webcam()
webcam.saveVideos()