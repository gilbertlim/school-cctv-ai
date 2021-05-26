import os
from queue import Queue

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ModuleNotFoundError as e:
    print(e)
    os.system("\n# Tracker : pip install watchdog")


class Target:
    q_video = Queue()
    q_json = Queue()

    def __init__(self, watch_path):
        self.dir = watch_path
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler,
                               self.dir,
                               recursive=True)
        self.observer.start()

        print('\n# Tracker : file tracker stared')
        try:
            while True:
                pass
        except:
            self.observer.stop()
            print("\n# Tracker : Error")
            self.observer.join()


class Handler(FileSystemEventHandler):
    # Override
    def on_moved(self, event):  # move or rename
        pass
        # print(event)

    def on_created(self, event):  # create
        if event.src_path.split('.')[-1] == 'mp4':
            Target.q_video.put(event.src_path)
        elif event.src_path.split('.')[-1] == 'json':
            Target.q_json.put(event.src_path)
        # print(Target.q.queue)

    def on_deleted(self, event):  # delete
        pass
        # print(event)

    def on_modified(self, event):  # modify
        pass
        # print(event)
