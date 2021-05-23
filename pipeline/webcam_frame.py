# import Library
import cv2
import os
import threading

from glob import glob
from queue import Queue

import numpy as np
import pandas as pd


# define constant variabes

BODY_PARTS_COCO = {1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle"}
# BODY_PARTS_COCO = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
#                    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
#                    10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
#                    15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

# POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
#                    [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]

protoFile_coco = '/Users/gilbert/Developer/Project/3_Convergence/School_CCTV_AI/pipeline/openpose/models/pose/coco/pose_deploy_linevec.prototxt'
weightsFile_coco = '/Users/gilbert/Developer/Project/3_Convergence/School_CCTV_AI/pipeline/openpose/models/pose/coco/pose_iter_440000.caffemodel'

weights_yolo = '/Users/gilbert/Developer/Project/3_Convergence/School_CCTV_AI/pipeline/yolo/yolov3-320.weights'
# weights_yolo = '/Users/gilbert/Downloads/yolov3-tiny.weights'

config_yolo = '/Users/gilbert/Developer/Project/3_Convergence/School_CCTV_AI/pipeline/yolo/yolov3-c1.cfg'
# config_yolo = '/Users/gilbert/Downloads/yolov3-tiny.cfg'

coco_names = '/Users/gilbert/Developer/Project/3_Convergence/School_CCTV_AI/pipeline/yolo/coco.names'

q_angle = Queue()
q_predict = Queue()


def yolo(frame): # YOLO V3
    # load network
    net = cv2.dnn.readNet(weights_yolo, config_yolo)

    with open(coco_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    img = frame
    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Calculate Object Coordinates
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    crop = []
    c = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == 'person':
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                c.append((x, y, x + w, y + h))
    crop.append(c)

    return frame, crop

def output_keypoints(frame, proto_file=protoFile_coco, weights_file=weightsFile_coco, threshold=0.1, model_name='COCO', BODY_PARTS=BODY_PARTS_COCO):
    # load network
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # input image size
    image_height = 368
    image_width = 368

    # Preprocessing
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

    # Input preprocessed blob to network
    net.setInput(input_blob)

    # Extracting joints data
    out = net.forward()

    out_height = out.shape[2]
    out_width = out.shape[3]

    # load original image size
    frame_height, frame_width = frame.shape[:2]

    # initiate point
    points = []

    print(f"\n============================== {model_name} Model ==============================")
    for i in BODY_PARTS.keys():
        # confidence map
        prob_map = out[0, i, :, :]

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

            points.append(x)
            points.append(y)

            print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    return frame, points

class Webcam:
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture('/Users/gilbert/Developer/Project/3_Convergence/School_CCTV_AI/pipeline/video.mp4') # test Video
        # self.cap = cv2.VideoCapture(0) # use webcam
        self.frame_size = (640, 480)

    def captureFrames(self):
        while True:
            ret, frame = self.cap.read()  # capture by 1 frame

            if not ret:
                break

            frame = cv2.flip(frame, 1) # horizontal flip
            frame = cv2.resize(frame, self.frame_size) # image resize

            # Object Detection
            y_frame, coordinates = yolo(frame)

            cropped_images = []
            cor_x, cor_y = 0, 0
            for coord in coordinates:
                for c in coord:
                    cropped_images.append(y_frame[c[1]:c[3], c[0]:c[2]])
                    cor_x = c[0]
                    cor_y = c[1]

            print('Detected # of person :', len(cropped_images))

            # Extract joints data
            keypoints = []
            for ci in cropped_images:
                o_frame, joints = output_keypoints(ci)

                for idx, j in enumerate(joints):
                    if (idx + 1) % 2 != 0:
                        x = j
                        y = joints[idx + 1]

                        # print(x, y)
                        cv2.circle(frame, (int(x) + cor_x, int(y) + cor_y), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

                keypoints.append(joints)
                cv2.imshow('frame', frame)

            q_angle.put(keypoints)
            print(q_angle.queue)

            # Streaming
            # cv2.imshow('frame', tar_frame)

            # Shutdown Webcam if press 'ESC'
            key = cv2.waitKey(25)
            if key == 27:
                break

        self.cap.release() # realse cap object


webcam = Webcam()
webcam.captureFrames()