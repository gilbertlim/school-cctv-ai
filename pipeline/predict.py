import threading

import numpy as np

import tensorflow as tf
from tensorflow import keras

np.set_printoptions(precision=4, suppress=True)
with tf.device('/cpu:0'):
    model = keras.models.load_model('./models/lstm_p1_walking_smoking_punching_running_kicking_angle.h5', compile = False)

classes = {0 : 'walking', 1 : 'smoking', 2: 'punching', 3 : 'running', 4 : 'kicking'}

class Predict(threading.Thread):
    def __init__(self, q_nparray):
        super().__init__()
        self.q_nparray = q_nparray

    def main(self):
        # print('q_nparray_length : ', len(self.q_nparray.queue))
        # print('q_nparray : ', q_nparray.queue)

        inputs = []
        nfpeople = 0

        if len(self.q_nparray.queue) >= 32:
            for i in range(32):
                inputs.append(self.q_nparray.get())
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

            with tf.device('/cpu:0'):
                output = model.predict(globals()['p_' + str(i)])
            output = np.argmax(output[0], axis=-1)
            output = classes[int(output)]
            predicted.append(output)

        # print('predicted', predicted)

        return predicted