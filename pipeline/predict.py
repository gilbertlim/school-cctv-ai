import threading

import numpy as np

import tensorflow as tf
from tensorflow import keras

np.set_printoptions(precision=4, suppress=True)
with tf.device('/cpu:0'):
    model_path = './models/lstm_1_june_16_50.h5'
    model = keras.models.load_model(model_path)

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

            nfpeople = []
            for i in inputs:
                nfpeople.append(len(i))
            nfpeople = max(nfpeople)
        # print('inputs', inputs)

        # 사람 수에 따른 변수 선언(동적 변수)
        for i in range(nfpeople):
            globals()['p_' + str(i)] = []

        # 사람별 데이터 분리(p_0, p_1, ..., p_N)
        for ip in inputs:
            for i in range(len(ip)):
                try:
                    if ip[i] is not None:
                        globals()['p_' +str(i)].append(ip[i])
                except KeyError as e:
                    print('KeyError', ip, e)

        predicted = []
        for i in range(nfpeople):
            globals()['p_' +str(i)] = np.array(globals()['p_' +str(i)])

            if len(globals()['p_' +str(i)]) == 32:
                globals()['p_' +str(i)] = globals()['p_' +str(i)].reshape(-1, 32, 8)

                with tf.device('/cpu:0'):
                    output = model.predict(globals()['p_' + str(i)])
                output = np.argmax(output[0], axis=-1)
                output = classes[int(output)]
                predicted.append(output)

                # print('predicted', predicted)
            else:
                del globals()['p_' + str(i)]

        return predicted