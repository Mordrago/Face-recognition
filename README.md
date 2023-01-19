# Face-recognition
An application using the PyQT library and OpenCv for face recognition. It is possible to add a new face and train the LBPHF algorithm to add a new face to the person database. 

# Note 1
haarcascade_frontalface_default.xml is a standard OpenCV library file. I added it to the project to make it easier to use the program. You can refer directly to the OpenCV library. 

# Note 2 
If cv.face.LBPHFaceRecognizer_create() does not work in addition to the OpenCV library install opencv-contrib-python

# Note 3
If you want to use a different camera it may cause you to use the serial library 

# Note 4
If you want the program to run more smoothly or have a constant camera view you can use Qthread

# Note 5
You can use other face recognition algorithms. The OpenCV library supports other algorithms, including those related to machine learning. 
