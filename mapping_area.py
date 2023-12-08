import cv2
import numpy as np
import cvzone
import pickle

drawing=False

try:
    with open("data_parkir", "rb") as f:
        data=pickle.load(f)
        polylines,area_names=data['polylines'],data['area_names']
except:
    area_names=[]
    polylines=[]

points=[]
current_name=" "

def draw(event,x,y,flags,param):
    global points,drawing
    drawing=True
    if event==cv2.EVENT_LBUTTONDOWN:
        points=[(x,y)]
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x,y))
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        current_name=input('area name:-')
        if current_name:
            area_names.append(current_name)
            polylines.append(np.array(points,np.int32))
cap = cv2.VideoCapture('easy1.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame=cv2.resize(frame,(1020,500))
    for i,polyline in enumerate(polylines):
        print(i)
        cv2.polylines(frame,[polyline],True,(0,0,255),2)
        cvzone.putTextRect(frame,f'{area_names[i]}',tuple(polyline[0]),1,1 )
    cv2.imshow('FRAME', frame)
    cv2.setMouseCallback('FRAME',draw)
    Key = cv2.waitKey(100) & 0xFF
    if Key==ord('s'):
        with open("data_parkir", "wb") as f:
            data={'polylines' : polylines, 'area_names':area_names}
            pickle.dump(data,f)
cap.release()
cv2.destroyAllWindows()