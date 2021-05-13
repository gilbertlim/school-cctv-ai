import cv2
import os

cap = cv2.VideoCapture(0)

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

print('frame_size = ', frame_size)
print('fps :', cap.get(5))


cnt = 0
target_frames = 32
v_path = './Videos/video.mp4'


# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 압축 알고리즘
out = cv2.VideoWriter(v_path, fourcc, 20.0, frame_size)


if not os.path.isdir('./Videos'):
    os.mkdir('Videos')

while True:
    ret, frame = cap.read()  # 1 프레임씩 캡처
    print(cnt)

    # frame = cv2.flip(frame, 0)
    frame = cv2.flip(frame, 1) # horizontal flip

    if not ret:
        break

    num = cnt // 32
    print(num)

    if cnt % 32 == 0:
        v_path = './Videos/video_' + str(num) + '.mp4'
        out = cv2.VideoWriter(v_path, fourcc, 20.0, frame_size)
        print(v_path)

    out.write(frame)

    cnt += 1

    cv2.imshow('frame', frame)

    key = cv2.waitKey(25)
    if key == 27 and cnt % 32 == 0:
        break  # ESC


cap.release()
out.release()
os.remove('./Videos/video.mp4')