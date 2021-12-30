# %%
import cv2
from datetime import datetime
import face_recognition as fr
import numpy as np
import os
import pandas as pd


# %% define functions 
def get_db_faces(db_path):
    """
    Encodes the images found in the specified path.

    return dict(name, encoding )
    """
    encoded = {}
    counter = 1
    items, images =  find_images(db_path)
    for image in images:
        fname = os.path.splitext(image) 

        face = fr.load_image_file(f'./FaceDB/{image}')
        print(f'encoding {counter}/{items} images...')
        encoding = fr.face_encodings(face)[0]
        encoded[fname[0]] = encoding
        counter += 1
    print('encoding done.')

    return encoded

def find_images(path):
    """
    stores the filenames of .jpg/.png images in a list.

    returns the number of images found and their filenames

    return int, list
    """
    items = 0
    images = []
    for directory in os.listdir(path):
        if directory.endswith('.jpg') or directory.endswith('.png'):
            items += 1
            images.append(directory)
    print(f'Found {items} images.')
    return  items, images

def encodeFace(img):
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25) # resize the image to 1/4 
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    encoding = fr.face_encodings(imgS)[0]
    return encoding
# blabla
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
            f.writelines(f'\n{name},{dtString}')
            print("Attendance recorded.")
# %% locate and encode
known_faces = get_db_faces('./FaceDB')
known_names = list(known_faces.keys())
known_encodings = list(known_faces.values())
print(known_names)




# %%  capture test image through webcam
    #change to VideoCapture(1) for desktops or laptops that uses external webcams
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25) # resize the image to 1/4 
    # imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    curFrameLoc = fr.face_locations(imgS)
    curFrameEnc = fr.face_encodings(imgS, curFrameLoc)

    names = []

    for unknown_encoding in curFrameEnc:
        name = 'Unknown'
        matches = fr.compare_faces(known_encodings, unknown_encoding)
        distances = fr.face_distance(known_encodings, unknown_encoding)
        
        match_index = np.argmin(distances)

        if matches[match_index]:
            name = known_names[match_index]
        names.append(name)

        for name, location in zip(names, curFrameLoc):
            markAttendance(name)

            # needs more planning here
            # if name == 'Unknown':
            #     now = datetime.now()
            #     uk_fname = now.strftime('%H:%M:%S %d-%m-%Y')
            #     cv2.imwrite(f'Unknowns/{uk_fname}.jpg', img)

            y1,x2,y2,x1 = location
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            
            # Draw a box around the face
            cv2.rectangle(img, (x1-20, y1-20), (x2+20, y2+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (x1-20, y2-15), (x2+20, y2+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (x1 -20, y2 + 15), font, 1.0, (255, 255, 255), 2)


        
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# %% Notes 
# Name unknown images with datetime
# cap.release()
# cv2.destroyAllWindows()
