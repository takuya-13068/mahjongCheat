#動画から学習画像を切り抜く処理
import cv2
import os
import sys

def save_frame_range(video_path, start_frame, stop_frame, step_frame,
                     dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    for n in range(start_frame, stop_frame, step_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
        else:
            return

#stand, hand, downを選択する
args = sys.argv
save_frame_range('data/video/'+args[1], 0, 3000, 10, 'data/'+args[2], args[3])
#300sec 
#python process.py (Videofilename) (foldername) (filename)
# ex) python process.py hand.mp4 hand img1