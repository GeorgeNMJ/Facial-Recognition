
Dependencies:
- cmake 
- dlib
- face-recognition
- numpy
- pandas
- matplotlib
- opencv-python
- streamlit

To run source code:
1. activate conda environment with the dependencies
2. streamlit run main.py

Issues:
- works only for laptops since it uses the cv2.VideoCapture(0) in its source code.
- open webcam sometimes does not work, ticking it off and on again will run it.
- attendance dataset does not update realtime.
