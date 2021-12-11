# %%
import cv2
import numpy as np
import face_recognition


# %% loading the images
imgElon = face_recognition.load_image_file('ImagesBasic/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Elon Test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


# %% locating the face
# locate the boundary where the face is
# returned value is a tuple (top, right, bottom, left)
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]

# draw a square using the boundaries in facecLoc
cv2.rectangle(imgElon,
        # start and end coordinates of the rectable
        (faceLoc[3], faceLoc[0]),
        (faceLoc[1], faceLoc[2]),
        # color and pixel of the rectangle
        (255,0,255), 2
    )

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,
        (faceLocTest[3], faceLocTest[0]),
        (faceLocTest[1], faceLocTest[2]),
        (255,0,255), 2
    )


# %%  comparing and finding distance between faces 
results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
cv2.putText( imgTest 
           , f'{results}: {round(faceDis[0], 2)}'
           , (50,50)
           , cv2.FONT_HERSHEY_COMPLEX
           , 1
           , (0,0,255)
           , 2
           )


# %% showing the images
cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)



# %%
cv2.destroyWindow('Elon Musk')
cv2.destroyWindow('Elon Test')


