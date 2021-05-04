import cv2 # pip3 install opencv-python
import xml.etree.ElementTree as et

class Preprocessing:
    def __init__(self, v_path):
        self.v_path = v_path
        self.xml_path = self.v_path[:-4]  + '.xml'
        self.cap = cv2.VideoCapture(self.v_path)  # VideoCapture 객체 생성

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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()  # 카메라 리소스 해제
        cv2.destroyAllWindows()  # 열린 창을 모두 닫음

    def parsingXML(self): # XML 파일 해석
        tree = et.parse(self.xml_path)
        root = tree.getroot()

        print('----- Parsing XML -----')
        # print(root.tag) # root tag name
        # print(root.attrib) # root attribute

        # event
        evnt = root.findall('event')
        evnt_startime = [e.findtext('starttime') for e in evnt]
        evnt_duration = [e.findtext('duration') for e in evnt]
        print('[ event ]')
        print('starttime :', evnt_startime) # event start time
        print('duration :', evnt_duration) # event duration
        print()

        # object
        objt =  root.findall('object')
        objt_name = [o.findtext('objectname') for o in objt]
        objt_frame = [o.find('position').findtext('keyframe') for o in objt]
        objt_x_pos = [i.text for i in root.iter('x')]
        objt_y_pos = [i.text for i in root.iter('y')]
        print('[ object ]')
        print('object name :', objt_name) # object name
        print('object frame :', objt_frame) # object frame
        print('object x pos :', objt_x_pos) # object x position
        print('object y pos :', objt_y_pos) # object y position
        print()

        # action
        print('[ action ]')
        act_name = []
        for o in objt:
            act_name.append([i.findtext('actionname') for i in o.findall('action')])
        print('act_name :', act_name) # action name

        act_start_frame = [[] for _ in range(len(objt))]
        act_end_frame = [[] for _ in range(len(objt))]
        for idx, o in enumerate(objt):
            for a in o.iter('action'):
                start = a.iter('start')
                end = a.iter('end')

                f_s = [s.text for s in start]
                act_start_frame[idx].append(f_s)

                f_e = [e.text for e in end]
                act_end_frame[idx].append(f_e)

        print('act_start_frame :', act_start_frame) # action start frame
        print('act_end_frame :', act_end_frame) # action end frame

    def cropVideos(self):
        pass

def definePath():
    # 절대 경로
    absolute_path = '/Users/gilbert/Developer/Project/3_Convergence/Development'
    # 비디오 파일 경로
    video_path = '/datasets/insidedoor_01/10-1/10-1_cam01_assault03_place07_night_spring.mp4'

    return absolute_path + video_path


def main():
    prep = Preprocessing(definePath()) # 인스턴스 생성

    prep.printVideoMeta() # 메타 데이터 출력
    # prep.readAndShowVideo() # 비디오 재생
    prep.parsingXML() # 라벨 출력


main()