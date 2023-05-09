import csv
import pprint
import pathlib
import cv2
import mediapipe as mp
from pathlib import Path
import sys

def editCSV(p, file, array, category):
    path = Path(p)
    isempty = path.stat().st_size == 0
    if isempty:
        with open(p, 'w') as f:
            field_name = ['label', 'file', 'data']
            writer = csv.DictWriter(f, fieldnames = field_name)
            writer.writeheader()
            writer.writerow({'label': category, 'file': file, 'data': array})

    else:
        with open(path, 'a') as f:
            field_name = ['label', 'file', 'data']
            writer = csv.DictWriter(f, fieldnames = field_name)
            writer.writerow({'label': category, 'file': file, 'data': array})

args = sys.argv
input_dir = 'data/'+args[1]
input_list = list(pathlib.Path(input_dir).glob('*.jpg'))

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# 画像を読み込んでpose estimation 処理を行う(配列)
data = []
for i in range(len(input_list)):
    err = False
    poseData = []
    file_name = str(input_list[i])
    img = cv2.imread(file_name)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            #print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            #cv2.circle(img, (cx, cy), 4, (255, 0, 0), cv2.FILLED)
            #print(cx, cy)
            if not(lm.visibility > 0.1 and cx > 0 and cx < 1080 and cy > 0 and cy < 1920):
                print("ERROR! file:"+file_name+" is not fully recognized")
                err = True
                break
            poseData += [cx,cy]
        poseData = " ".join(map(str,poseData))
        if not err:
            # stand:0 hand:1, down:2
            if args[1] == 'stand':
                editCSV('data/main.csv', file_name, poseData, 0)
            elif args[1] == 'hand':
                editCSV('data/main.csv', file_name, poseData, 1)
            elif args[1] == 'down':
                editCSV('data/main.csv', file_name, poseData, 2)
            elif args[1] == 'val':
                editCSV('data/val.csv', file_name, poseData, 100)
