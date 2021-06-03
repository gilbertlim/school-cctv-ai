#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install opencv-python


# In[34]:


import numpy as np
import cv2

# cv2.startWindowThread()
cap = cv2.VideoCapture(0)


# In[35]:


if cap.isOpened():
    print('Current position(milisec) :', cap.get(0))
    print('Index of the frame :', cap.get(1))
    print('Relative position of the video :', cap.get(2))
    print('Width :', cap.get(3))
    print('Height :', cap.get(4))
    print('Frame rate :', cap.get(5))
    print('Number of frames :', cap.get(7))


# In[39]:


while(True):
  # 프레임 읽기
  ret, frame = cap.read()

  # 프레임 디스플레이
  cv2.imshow('frame',frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    # 루프 break
    # 비디오는 반드시 하이라이트 되어있어야 함
    break

cap.release()
cv2.destroyAllWindows()
# 맥에선 필요함
# cv2.waitKey(1)


# In[5]:


# # turn to greyscale:
# frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
# # apply threshold. all pixels with a level larger than 80 are shown in white. the others are shown in black:
# ret,frame = cv2.threshold(frame,80,255,cv2.THRESH_BINARY)


# In[30]:


# Human detection
import numpy as np
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# 웹캠 비디오 스트림
cap = cv2.VideoCapture(0)

# output은 output.avi로 쓰일것
out = cv2.VideoWriter('output.avi',
                     cv2.VideoWriter_fourcc(*'MJPG'),
                      15.,
                      (640,480))


# In[31]:


while(True):
    # frame by frame 으로 캡쳐
    ret, frame = cap.read()
    
    # 더 빠른 detection을 위해 리사이징
    frame = cv2.resize(frame, (640, 480))
    # 더 빠른 detection을 위해 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 이미지 안 사람 detection
    # 감지한 오브젝트를 return
    boxes, weights = hog.detectMultiScale(frame, winStride = (8,8) )
    
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    
    for (xA, yA, xB, yB) in boxes:
        # 칼라 이미지로 감지된 박스 디스플레이
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
        
    out.write(frame.astype('uint8'))
    # 결과 frame 디스플레이
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# 다 끝난 후 캡쳐한 것 release
cap.release()
# output release
out.release()
# 윈도우 닫기
cv2.destroyAllWindows()
cv2.waitKey(1)


# In[ ]:





# In[44]:


# 다운 받은 영상 내 감지
path = 'C:/Users/user/10-1_cam01_assault03_place07_night_spring.mp4' # 비디오 파일 경로
cap = cv2.VideoCapture(path) # VideoCapture 객체 생성


# In[46]:


# Human detection
import numpy as np
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# 웹캠 비디오 스트림
cap = cv2.VideoCapture(path)

# output은 output.avi로 쓰일것
out = cv2.VideoWriter('output.avi',
                     cv2.VideoWriter_fourcc(*'MJPG'),
                      15.,
                      (640,480))
while(True):
    # frame by frame 으로 캡쳐
    ret, frame = cap.read()
    
    # 더 빠른 detection을 위해 리사이징
    frame = cv2.resize(frame, (640, 480))
    # 더 빠른 detection을 위해 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 이미지 안 사람 detection
    # 감지한 오브젝트를 return
    boxes, weights = hog.detectMultiScale(frame, winStride = (8,8) )
    
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    
    for (xA, yA, xB, yB) in boxes:
        # 칼라 이미지로 감지된 박스 디스플레이
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
        
    out.write(frame.astype('uint8'))
    # 결과 frame 디스플레이
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# 다 끝난 후 캡쳐한 것 release
cap.release()
# output release
out.release()
# 윈도우 닫기
cv2.destroyAllWindows()
cv2.waitKey(1)


# In[ ]:




