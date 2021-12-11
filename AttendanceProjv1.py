# %%
import cv2 
import numpy as np
import face_recognition as fr
import os


# %% loading images
# imgTrain will contain the list of known faces
# imgTest will contain the list of unknown face
path = '/home/georgem/Documents/python/Facial-recognition/'
os.chdir(path)

imgTrain = []
classNames = []
for directory in os.listdir('ImagesAttendance'):
    classNames.append(directory.split('.')[0])
    imgTrain.append(fr.load_image_file(f'ImagesAttendance/{directory}'))
    imgTrain[-1] = cv2.cvtColor(imgTrain[-1], cv2.COLOR_BGR2RGB)


# %% define functions 
def locateFace(images):
    face_location = []
    for image in images:
        location = fr.face_locations(image)[0]
        face_location.append(location)
    return face_location
def encodeFace(images):
    face_encode = []
    for image in images:
        encode = fr.face_encodings(image)[0]
        face_encode.append(encode)
    return face_encode
def rectangle(images, locations):
    for image, face_location in zip(images,locations):
        cv2.rectangle(image,
            (face_location[3], face_location[0]),
            (face_location[1], face_location[2]),
            (255,0,255), 2
    )


# %%
faceLocsTrain = locateFace(imgTrain)
encodingsTrain = encodeFace(imgTrain)
rectangle(imgTrain, faceLocsTrain)


# %% load test images
imgTest = fr.load_image_file('ImagesBasic/Margot Test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB) 


# %%
faceLocTest = locateFace([imgTest])[0]
encodingTest = encodeFace([imgTest])[0]
rectangle([imgTest], [faceLocTest])

# %% comparing results

results = fr.compare_faces(encodingsTrain, encodingTest)
faceDis = fr.face_distance(encodingsTrain, encodingTest)

y0, dy = 185,50 
text = ''
for name, result in zip(classNames, results):
    text += f'{name}: {result}\n' 
text+= f'Distance: {round(faceDis[0], 2)}'
for i, line in enumerate(text.split('\n')):
    y = y0 + i*dy
    cv2.putText(imgTest, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)

print(text)


# %% show images
# for image_name, image in zip(classNames, imgTrain):
    # cv2.imshow(image_name, image)
cv2.imshow('Test Image', imgTest)
cv2.waitKey(0)


# %% close window
# for image_name in classNames:
    # cv2.destroyWindow(image_name)
cv2.destroyWindow('Test Image')

# %% save image result
cv2.imwrite('ImagesResult/Test-Jennie.jpg', imgTest)
