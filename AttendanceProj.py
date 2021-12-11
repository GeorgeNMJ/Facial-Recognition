# %%
import cv2 
import numpy as np
import face_recognition as fr
import os
from datetime import datetime

# %% loading images
# imgTrain will contain the list of known faces
# imgTest will contain the list of unknown face
path = '/home/georgem/Documents/python/Facial-recognition/'
os.chdir(path)

imgTrain = []
classNames = []
for directory in os.listdir('ImagesAttendance'):
    classNames.append(os.path.splitext(directory)[0])
    imgTrain.append(fr.load_image_file(f'ImagesAttendance/{directory}'))
    imgTrain[-1] = cv2.cvtColor(imgTrain[-1], cv2.COLOR_BGR2RGB)

# %% define functions 

def encodeFace(images):
    face_encode = []
    for image in images:
        encode = fr.face_encodings(image)[0]
        face_encode.append(encode)
    return face_encode

def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S %d-%m-%Y')
            f.writelines(f'{name},{dtString}\n')


# %% locate and encode

print('Encoding images...')
encodingsTrain = encodeFace(imgTrain)
print('Encoding complete.')

# %%  capture test image through webcam
cap = cv2.VideoCapture(0)

# a video is essentially an image in a slideshow 
# so we call the variables currFrame
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25) # resize the image to 1/4 
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    currFrameLocation = fr.face_locations(imgS)
    currFrameEncode = fr.face_encodings(imgS, currFrameLocation)
    
    for faceEncode, faceLoc in zip(currFrameEncode, currFrameLocation):
        matches = fr.compare_faces(encodingsTrain, faceEncode)
        distances = fr.face_distance(encodingsTrain, faceEncode)
        print(distances)

        matchIndex = np.argmin(distances)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            markAttendance(name)
            print('Attendance recorded.')
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText( img
                       , name
                       , (x1+6, y2-6)
                       , cv2.FONT_HERSHEY_COMPLEX, 1
                       , (255,255,255), 2
                       )

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# %% 
cap.release()
cv2.destroyAllWindows()

