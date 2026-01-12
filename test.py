from sklearn.neighbors import KNeighborsClassifier
import cv2 
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak (str1):
     speak=Dispatch(("SAPI.SpVoice"))
     speak.Speak(str1)
     




os.makedirs("data",exist_ok=True)


video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH,600)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,800)

facedetect  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


with open ('data/names.pkl','rb') as f:
        LABLES=pickle.load(f)

with open ('data/faces_data.pkl','rb') as f:
        FACES=pickle.load(f)


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABLES)
imgBackground=cv2.imread(r"C:\Users\MADHU\Desktop\attendence_sys\bckgrund.png")
imgBackground= cv2.resize(imgBackground,(800,800))

COL_NAMES=['NAME','DATE','TIME']




while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h,x:x+w,:]
        resized_img=cv2.resize(crop_img,(50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        filename="Attendance/Attendance_"+date+".csv"
        exist=os.path.isfile(filename) and os.path.getsize(filename)>0
        
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame,str(output[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)
        attendence=[str(output[0]),str(date),str(timestamp)]

        
    h,w,_=frame.shape
    imgBackground[120:120+h,50:50+w] = frame
    
    

    cv2.imshow("frame",imgBackground)
    K=cv2.waitKey(1)
    if K==ord('o'):
          speak("Attendance taken ....")
          time.sleep(5)
          if exist:
             with open ("Attendance/Attendance_"+date+".csv","+a") as csvfile:
                    writer=csv.writer(csvfile)  
                    writer.writerow(attendence)
             csvfile.close()
             
          else:
               with open ("Attendance/Attendance_"+date+".csv","+a") as csvfile:
                    writer=csv.writer(csvfile)  
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendence)
               csvfile.close()   
    if K==ord('q'):
        break
video.release()
cv2.destroyAllWindows() 










