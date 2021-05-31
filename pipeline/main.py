import os
from threading import Thread
from queue import Queue

from preprocessing import Preprocessing
from extract_json import Extractjson
from predict import Predict
from tracker import Target


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)

        self._return = None
    def run(self):
        # print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class Main:
    def __init__(self):
        self.root = '/home/ubuntu/CCTV/'
        self.v_path = self.root + 'videos/'
        self.j_path = self.root + 'json/'

        self.q_video = Queue()
        self.q_nparray = Queue()
        self.q_predict = Queue()

    def makeDirectory(self):
        if not os.path.isdir(self.root + 'json'):
            os.mkdir(self.root + 'json')
        if not os.path.isdir(self.root + 'videos'):
            os.mkdir(self.root + 'videos')
        if not os.path.isdir(self.root + 'old_videos'):
            os.mkdir(self.root + 'old_videos')

    def findProcess(self):
        import psutil
        for proc in psutil.process_iter():
            try:
                processName = proc.name() # read process name
                if processName[:39] == "./build/examples/openpose/openpose.bin":
                    return 'running'
                else:
                    return 'empty'
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        # print('동일 프로세스 확인 완료....')

    def main(self):
        self.makeDirectory() # Make directories

        # Thread 0 : Video file tracking
        print('\n# Thread 0 : Video file tracking')
        video_tracker = Target(self.v_path)
        t0 = Thread(target=video_tracker.run)
        t0.daemon = True
        t0.start()

        # Thread 1 : JSON file tracking
        print('\n# Thread 1 : JSON file tracking')
        json_tracker = Target(self.j_path)
        t1 = Thread(target=json_tracker.run)
        t1.daemon = True
        t1.start()

        while True:
            q_len_n = len(video_tracker.q_video.queue)

            if q_len_n == 2:
                cur_process = self.findProcess()

                if cur_process == 'empty':
                    v_list = video_tracker.q_video.get()

                    # Thread 2 : Extract json from videos
                    ext_json = Extractjson('../videos' + v_list[v_list.rfind('/'):])
                    t2 = Thread(target=ext_json.main)
                    t2.daemon = True
                    t2.start()

                else:
                    print('\n openpose is already running')

            q_len_j = len(json_tracker.q_json.queue)
            if q_len_j == 1:
                j_list = json_tracker.q_json.get()

                # Thread 3 : Convert json to Numpy Array
                prep = Preprocessing(j_list)
                twrv_numpy = ThreadWithReturnValue(target=prep.main)
                twrv_numpy.start()

                if twrv_numpy.join() is not None:
                    self.q_nparray.put(twrv_numpy.join())
                if len(self.q_nparray.queue) >= 1:
                    print('\n# Thread 3 : nparray length : ', len(self.q_nparray.queue))

            # Thread 4 : Predict
            predict = Predict(self.q_nparray)
            twrv_predict = ThreadWithReturnValue(target=predict.main)
            twrv_predict.start()

            if twrv_predict.join() is not None:
                if len(twrv_predict.join()) != 0:
                    self.q_predict.put(twrv_predict.join())

            if len(self.q_predict.queue) >= 1:
                predicted = self.q_predict.get()
                print('\n# Thread 4 : predicted', predicted)


main = Main()
main.main()