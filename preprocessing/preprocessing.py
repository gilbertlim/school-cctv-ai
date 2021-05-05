import cv2 # pip3 install opencv-python
import os
from moviepy.editor import *  # pip3 install moviepy
from datetime import datetime, timedelta
import xml.etree.ElementTree as et

class Preprocessing:
    def __init__(self, v_path):
        self.v_path = v_path
        self.xml_path = self.v_path[:-4]  + '.xml'
        self.cap = cv2.VideoCapture(self.v_path)  # VideoCapture 객체 생성

        self.tree = et.parse(self.xml_path) # xml 객체 파싱 객체 생성
        self.root = self.tree.getroot() # 루트 초기화

        self.evnt = self.root.findall('event')
        self.evnt_startime = [e.findtext('starttime') for e in self.evnt][0].split('.')[0]
        self.evnt_startime = datetime.strptime(self.evnt_startime, '%H:%M:%S').time()
        self.evnt_startime_split = str(self.evnt_startime).split(':')
        self.evnt_duration = [e.findtext('duration') for e in self.evnt][0].split('.')[0]
        self.evnt_duration = datetime.strptime(self.evnt_duration, '%H:%M:%S').time()
        self.evnt_duration_split = str(self.evnt_duration).split(':')
        self.evnt_endtime = timedelta(hours=int(self.evnt_startime_split[0]), minutes=int(self.evnt_startime_split[1]), seconds=int(self.evnt_startime_split[2])) + \
                            timedelta(hours=int(self.evnt_duration_split[0]), minutes=int(self.evnt_duration_split[1]), seconds=int(self.evnt_duration_split[2]))

        self.objt =  self.root.findall('object')
        self.objt_name = [o.findtext('objectname') for o in self.objt]
        self.objt_frame = [o.find('position').findtext('keyframe') for o in self.objt]
        self.objt_x_pos = [int(i.text) for i in self.root.iter('x')]
        self.objt_y_pos = [int(i.text) for i in self.root.iter('y')]

        self.act_start_frame = [[] for _ in range(len(self.objt))]
        self.act_end_frame = [[] for _ in range(len(self.objt))]

    def printVideoMeta(self): # 비디오 메타 데이터 출력
        if self.cap.isOpened(): # VideoCapture 객체가 정의되어 있다면
            print('----- Video Meta data -----')
            print('Current position(milisec) :', self.cap.get(0))
            print('Index of the frame :', self.cap.get(1))
            print('Relative position of the video :', self.cap.get(2))
            print('Width :', self.cap.get(3))
            print('Height :', self.cap.get(4))
            print('Frame rate :', self.cap.get(5))
            print('Number of frames :', self.cap.get(7))
            print()
        else:
            print('VideoCapture is not defined')

    def readAndShowVideo(self): # 비디오 재생
        while True:
            ret, frame = self.cap.read()  # frame을 하나씩 읽음

            cv2.imshow('video', frame)  # frame 출력

            if cv2.waitKey(1) & 0xFF == ord('q'): # q 누르면 종료
                break
        self.cap.release()  # 카메라 리소스 해제
        cv2.destroyAllWindows()  # 열린 창을 모두 닫음

    def parsingXML(self): # XML 파일 해석
        print('----- Parsing XML -----')
        # print(root.tag) # root tag name
        # print(root.attrib) # root attribute

        # event
        print('[ event ]')
        print('starttime :', self.evnt_startime) # event start time
        print('duration :', self.evnt_duration) # event duration
        print('endtime :', self.evnt_endtime) # end time
        print()

        # object
        print('[ object ]')
        print('object name :', self.objt_name) # object name
        print('object frame :', self.objt_frame) # object frame
        print('object x pos :', self.objt_x_pos) # object x position
        print('object y pos :', self.objt_y_pos) # object y position
        print()

        # action
        print('[ action ]')
        act_name = []
        for o in self.objt:
            act_name.append([i.findtext('actionname') for i in o.findall('action')])
        print('act_name :', act_name) # action name

        for idx, o in enumerate(self.objt):
            for a in o.iter('action'):
                start = a.iter('start')
                end = a.iter('end')

                f_s = [s.text for s in start]
                self.act_start_frame[idx].append(f_s)

                f_e = [e.text for e in end]
                self.act_end_frame[idx].append(f_e)

        print('act_start_frame :', self.act_start_frame) # action start frame
        print('act_end_frame :', self.act_end_frame) # action end frame
        print()

    def cropVideos(self):
        print('----- Crop Videos -----')
        print('Trim and Resize Videos...')

        # Resize
        clip = (VideoFileClip(self.v_path,
                              target_resolution=(720, 1280),
                              resize_algorithm='lanczos',
                              audio=False)
                .subclip(str(self.evnt_startime), str(self.evnt_endtime))
                .crop()
                )

        # Clipping
        new_path = self.v_path.split('.')[0] + '_clipped(720p).mp4'
        if not os.path.isfile(new_path):
            clip.write_videofile(new_path)
        print('video already modified')


        self.cap = cv2.VideoCapture(new_path)  # VideoCapture 객체 생성

        while True:
            ret, frame = self.cap.read()  # frame을 하나씩 읽음

            p1_pos = (int(self.objt_x_pos[0] / 3), int(self.objt_y_pos[0] / 3)) # 3840 X 2160 -> 1280 X 720
            p2_pos = (int(self.objt_x_pos[1] / 3), int(self.objt_y_pos[1] / 3))
            cv2.line(frame, p1_pos, p1_pos, (255, 0, 0), 5)
            cv2.line(frame, p2_pos, p2_pos, (0, 0, 255), 5)

            cv2.imshow('video', frame)  # frame 출력

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()  # 카메라 리소스 해제
        cv2.destroyAllWindows()  # 열린 창을 모두 닫음


def definePath():
    # 절대 경로
    absolute_path = '/Users/gilbert/Developer/Project/3_Convergence/Dev_AI'
    # 비디오 파일 경로
    video_path = '/datasets/insidedoor_01/10-1/10-1_cam01_assault03_place07_night_spring.mp4'

    return absolute_path + video_path


def main():
    prep = Preprocessing(definePath()) # 인스턴스 생성

    prep.printVideoMeta() # 메타 데이터 출력
    prep.parsingXML() # Parsing Annotation

    # prep.readAndShowVideo() # 비디오 재생

    prep.cropVideos()

main()