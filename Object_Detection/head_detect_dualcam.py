# head_detect_dualcam.py
import cv2 as cv
import numpy as np
import time
import os

# model files (local linux paths)
MODEL = "frozen_inference_graph.pb"
CONFIG = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
LABELS = "label.txt"

# load labels
with open(LABELS, 'rt') as f:
    classLabels = f.read().rstrip('\n').split('\n')

# load network (OpenCV DNN)
net = cv.dnn.readNet(MODEL, CONFIG)
model = cv.dnn_DetectionModel(net)
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# open two cameras (use correct indices or /dev/videoX mapping)
cap0 = cv.VideoCapture(0)      # first camera -> /dev/video0
cap1 = cv.VideoCapture(2)      # second camera -> /dev/video2 (change if needed)

# set desired resolution (optional, helps performance)
cap0.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap0.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap1.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

if not cap0.isOpened() or not cap1.isOpened():
    print("Error: One or both cameras not opened. Check indices (/dev/video*).")
    exit(1)

font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 0.6

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        print("Camera read error")
        break

    # detect on frame0
    classIds0, confs0, boxes0 = model.detect(frame0, confThreshold=0.55)
    if len(classIds0) != 0:
        for classId, conf, box in zip(classIds0.flatten(), confs0.flatten(), boxes0):
            idx = int(classId) - 1
            label = classLabels[idx] if 0 <= idx < len(classLabels) else "Object"
            cv.rectangle(frame0, box, (255, 0, 0), 2)
            cv.putText(frame0, f"{label}:{conf:.2f}", (box[0]+5, box[1]+20), font, font_scale, (0,255,0), 1)

    # detect on frame1
    classIds1, confs1, boxes1 = model.detect(frame1, confThreshold=0.55)
    if len(classIds1) != 0:
        for classId, conf, box in zip(classIds1.flatten(), confs1.flatten(), boxes1):
            idx = int(classId) - 1
            label = classLabels[idx] if 0 <= idx < len(classLabels) else "Object"
            cv.rectangle(frame1, box, (255, 0, 0), 2)
            cv.putText(frame1, f"{label}:{conf:.2f}", (box[0]+5, box[1]+20), font, font_scale, (0,255,0), 1)

    # resize to same height and concat side-by-side
    h = 480
    frame0 = cv.resize(frame0, (int(frame0.shape[1]*h/frame0.shape[0]), h))
    frame1 = cv.resize(frame1, (int(frame1.shape[1]*h/frame1.shape[0]), h))
    combined = np.hstack((frame0, frame1))

    cv.imshow("Dual Camera Detection (press q to quit)", combined)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cv.destroyAllWindows()
