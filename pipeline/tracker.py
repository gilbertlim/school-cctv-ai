import os
import threading
import time

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ModuleNotFoundError as e:
    print(e)
    os.system("pip install watchdog")


class Target(threading.Thread):
    def __init__(self, watch_path):
        super().__init__()
        self.dir = watch_path
        self.observer = Observer()  # observer객체를 만듦

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler,
                               self.dir,
                               recursive=True)
        self.observer.start()

        try:
            while True:
                time.sleep(0.1)
        except:
            self.observer.stop()
            print("Error")
            self.observer.join()


class Handler(FileSystemEventHandler):
    # Override
    def on_moved(self, event):  # move or rename
        pass
        # print(event)

    def on_created(self, event):  # create
        print(event.src_path)
        # return event.src_path

    def on_deleted(self, event):  # delete
        pass
        # print(event)

    def on_modified(self, event):  # modify
        pass
        # print(event)


# path = './Videos/'
# w = Target(path)
# w.run()

