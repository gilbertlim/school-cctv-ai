import os
import threading

class Extractjson(threading.Thread):
    def __init__(self, v_path):
        super().__init__()
        self.v_path = v_path

    def main(self):  # openpose
        print('\n# Thread 2 : Exracting json... from ' + self.v_path + '\n')
        os.system(
            'cd openpose && ./build/examples/openpose/openpose.bin --video ' + self.v_path + ' --write_json ../json/ --display 0 --render_pose 0 --model_pose COCO')