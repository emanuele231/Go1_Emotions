*INSTALLATION GUIDE*

**EMOTION VISION SYSTEM**
In this guide, we provide to install a emotion vision system for UniTree Go1; 
this EVS have 3 Recongnition-Systems in it:
- Facial Recognition 
- Emotions Recognition
- Posture and body Recognition

Step 0: Enter in (venv) mode:
- source ~/unitree_sdk2_python/venv/bin/activate

First is the Facial Recognition *Haar-Cascades*:

1- First of all, you have to install *OpenCV 4.9.0.80*
pip install opencv-python==4.9.0.80 opencv-contrib-python==4.9.0.80

2- Move in OpenCV package
cd ~/unitree_sdk2_python/venv/lib/python3.10/site-packages/cv2/data/

3- Install all the dependencies Haar-Cascades
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_profileface.xml
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml

**Haar-Cascades is now installed**

Second Step: Emotions-Recognition *DeepFace*:

1- Install DeepFace
pip install deepface

2- DeepFace needs *tf-keras* to work with the recently versions of TensorFlow
pip install tf-keras==2.15.0
(Don't mind the Warning!)

**DeepFace is now installed**
*Notes that Haar-Cascades is necessary for DeepFace for the recognition of faces

Last Step: Posture and Body recognition *MediaPipe*:

1- It's important, on the top, install this dependencies for avoid conflicts:
pip install numpy==1.26.4 protobuf==4.25.3 tensorflow==2.15.0 mediapipe==0.10.14
This is called "Dependencies Hell" because every version depends of an other dependency version

**Now the EVS is ready!**

with - python3 ... you can now try all the recognitions!

