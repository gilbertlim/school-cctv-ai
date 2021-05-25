import math
import json
import numpy as np
import threading

class Preprocessing(threading.Thread):
    def __init__(self, json_paths):
        super().__init__()
        self.json_paths = json_paths

    def extractJoints(self):
        result = []

        with open(self.json_paths, "r") as json_file:
            # joint data 추출부
            json_object = json.load(json_file)
            json_array = json_object.get('people')  # joint 값 추출

            for j in json_array:
                keypoint = j.get('pose_keypoints_2d')
                for i in range(len(keypoint)):  # x,y 좌표 뒤에 위치한 확률값 제외하고 가져오기
                    if i % 3 != 2:
                        result.append(keypoint[i])
                    else:
                        pass

        nfpeople = len(json_array)

        return result, nfpeople

    def extractAngle(self, result, nfpeople):
        raw = []
        for nfp in range(nfpeople):
            raw.append(result[(36 * nfp): (36 + (nfp * 36))])

        def calculateAngle(a, j, c):
            a = np.array(a)
            j = np.array(j)
            c = np.array(c)

            a_j = a - j
            c_j = c - j

            th_a_j = math.atan2(a_j[1], a_j[0])
            th_c_j = math.atan2(c_j[1], c_j[0])

            return th_a_j - th_c_j

        angle = []
        for r in raw:
            neck = (r[2], r[3])
            r_shoulder = (r[4], r[5])
            r_elbow = (r[6], r[7])
            r_wrist = (r[8], r[9])
            r_hip = (r[16], r[17])
            r_knee = (r[18], r[19])
            r_ankle = (r[20], r[21])

            l_shoulder = (r[10], r[11])
            l_elbow = (r[12], r[13])
            l_wrist = (r[14], r[15])
            l_hip = (r[22], r[23])
            l_knee = (r[24], r[25])
            l_ankle = (r[26], r[27])

            ang_r_shoulder = calculateAngle(neck, r_shoulder, r_elbow)
            ang_r_elbow = calculateAngle(r_shoulder, r_elbow, r_wrist)
            ang_r_hip = calculateAngle(neck, r_hip, r_knee)
            ang_r_knee = calculateAngle(r_hip, r_knee, r_ankle)

            ang_l_shoulder = calculateAngle(neck, l_shoulder, l_elbow)
            ang_l_elbow = calculateAngle(l_shoulder, l_elbow, l_wrist)
            ang_l_hip = calculateAngle(neck, l_hip, l_knee)
            ang_l_knee = calculateAngle(l_hip, l_knee, l_ankle)

            angle.append([ang_r_shoulder, ang_r_elbow, ang_r_hip, ang_r_knee, ang_l_shoulder, ang_l_elbow, ang_l_hip, ang_l_knee])

        return angle

    def main(self):
        result, nfpeople = self.extractJoints()

        angle = self.extractAngle(result, nfpeople)

        return angle