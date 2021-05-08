import cv2 # pip3 install opencv-python
from moviepy.editor import *  # pip3 install moviepy
from datetime import datetime, timedelta # 시간 관련 계산을 위한 라이브러리
import xml.etree.ElementTree as et # xml 파일을 읽기 위한 라이브러리
import glob # 폴더 내 파일 리스트를 읽기 위한 라이브러리
import os

class Preprocessing:
    def __init__(self, v_path):
        self.v_path = v_path
        self.xml_path = self.v_path[:-4]  + '.xml'
        self.cap = cv2.VideoCapture(self.v_path)  # VideoCapture 객체 생성

        self.tree = et.parse(self.xml_path) # xml 객체 파싱 객체 생성
        self.root = self.tree.getroot() # 루트 초기화

        # parsing XML
        self.evnt = self.root.findall('event')
        self.evnt_startime = [e.findtext('starttime') for e in self.evnt][0].split('.')[0]
        self.evnt_startime = datetime.strptime(self.evnt_startime, '%H:%M:%S').time()
        self.evnt_startime_split = str(self.evnt_startime).split(':')
        self.evnt_duration = [e.findtext('duration') for e in self.evnt][0].split('.')[0]
        self.evnt_duration = datetime.strptime(self.evnt_duration, '%H:%M:%S').time()
        self.evnt_duration_split = str(self.evnt_duration).split(':')
        self.evnt_endtime = timedelta(hours=int(self.evnt_startime_split[0]), minutes=int(self.evnt_startime_split[1]), seconds=int(self.evnt_startime_split[2])) + \
                            timedelta(hours=int(self.evnt_duration_split[0]), minutes=int(self.evnt_duration_split[1]), seconds=int(self.evnt_duration_split[2]))

        # self.objt =  self.root.findall('object')
        # self.objt_name = [o.findtext('objectname') for o in self.objt]
        # self.objt_frame = [o.find('position').findtext('keyframe') for o in self.objt]
        # self.objt_x_pos = [int(i.text) for i in self.root.iter('x')]
        # self.objt_y_pos = [int(i.text) for i in self.root.iter('y')]

        self.act_name = [i.text for i in self.root.iter('actionname')]
        self.act_s_e = []

        for a in self.root.iter('action'):
            se = []
            frames = a.iter('frame')
            for f in frames:
                s, e = f.findtext('start'), f.findtext('end')
                se.append((s, e))
            self.act_s_e.append(se)

        # print(self.act_s_e)

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

    def printParsedData(self): # XML 파일 해석
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
        # print('[ object ]')
        # print('object name :', self.objt_name) # object name
        # print('object frame :', self.objt_frame) # object frame
        # print('object x pos :', self.objt_x_pos) # object x position
        # print('object y pos :', self.objt_y_pos) # object y position
        # print()

        # action
        print('[ action ]')
        print('act_name :', self.act_name) # action name

        print('act_frames(start, end) :' , self.act_s_e)
        print()

    def trimmingVideoEvent(self): # event가 존재하는 부분만 clipping
        print('Trim and Resize Videos by event...')

        clip = (VideoFileClip(self.v_path,
                              target_resolution=(720, 1280), # Resize
                              resize_algorithm='lanczos',
                              audio=False)
                .subclip(str(self.evnt_startime), str(self.evnt_endtime)) # trim
                # .crop()
                )

        # Clipping
        new_path = self.v_path.split('.')[0] + '_clipped_event(720p).mp4'

        if not os.path.isfile(new_path):
            clip.write_videofile(new_path)
        print('-> video already modified')
        print()

    def trimmingVideoAction(self):
        print('Trim and Resize Videos by action...')
        print()

        cnt = 1
        for idx, an in enumerate(self.act_name):
            for s, e in self.act_s_e[idx]:
                # print(idx, s, e)

                width, height = 640, 480

                clip = (VideoFileClip(self.v_path,
                                      target_resolution=(height, width),  # Resize
                                      resize_algorithm='lanczos',
                                      audio=False)
                        .subclip(int(s)/30, int(e)/30)  # trim
                        # .crop()
                        )

                # Clipping
                dir_clip = '/Users/gilbert/Developer/Project/3_Convergence/Dev_AI/action_clips/'
                # dir_clip = 'C:\\Users\\1672g\\Downloads\\datasets\\action_clips\\' # window OS example

                if not os.path.isdir(dir_clip):
                    os.mkdir(dir_clip)

                new_path = dir_clip + self.v_path.split('/')[-3] + '_' + self.v_path.split('/')[-1][:-4] + '_Clipped_' + an + '_' + s + '_' + e + '(' + str(height) + 'p)' + '.mp4'
                # new_path = dir_clip + self.v_path.split('/')[-3] + '_' + self.v_path.split('\\')[-1][:-4] + '_Clipped_' + an + '_' + s + '_' + e + '(' + str(height) + 'p)' + '.mp4' # window OS example

                if not os.path.isfile(new_path):
                    clip.write_videofile(new_path)
                else:
                    print('-> video already modified...{}' .format(cnt))
                    print()

                cnt += 1


def definePath():
    # 절대 경로
    absolute_path = '/Users/gilbert/Developer/Project/3_Convergence/Dev_AI/datasets/'
    # absolute_path = 'C:\\Users\\1672g\\Downloads\\datasets\\' # window OS example

    root_dir = glob.glob(absolute_path + '*')

    sub_dir = []
    for rd in root_dir:
        sub_dir.append(glob.glob(rd +'/*'))
        # sub_dir.append(glob.glob(rd +'\\*')) # window OS example

    videos = []
    for video_xml in sub_dir:
        for vx in video_xml:
            videos += glob.glob(vx + '/*')
            # videos += glob.glob(vx + '\\*') # window OS example

    videos = [v for v in videos if v.endswith('spring.mp4')]

    return videos


def main():
    video_path = definePath()
    for v in video_path:
        prep = Preprocessing(v) # 인스턴스 생성

        prep.trimmingVideoAction() # 행동이 있는 부분을 모두 추출하여 저장
        # prep.trimmingVideoEvent() # 이벤트가 있는 부분만 추출하여 저장


main()


# prep = Preprocessing('/Users/gilbert/Developer/Project/3_Convergence/Dev_AI/datasets/insidedoor_01/10-1/10-1_cam03_assault03_place07_night_spring.mp4')

# prep.readAndShowVideo() # 비디오 재생
# prep.printVideoMeta() # 메타 데이터 출력
# prep.printParsedData() # Parsing Annotation