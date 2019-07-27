import mtcnn
import numpy as np  
import pandas as pd  
import cv2  
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from keras.models import load_model
from PIL import Image


#Extracting faces from an image using MTCNN and always remember that MTCNN works 
# on only 160*160 pixel size

def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    
    # extract the bounding box from all faces and making a list of arrays

    all_faces_array = []
    for faces_arr in range(len(results)):

        x1, y1, width, height = results[faces_arr]['box']

        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        all_faces_array.append(face_array)

    return all_faces_array


#Loading pre-trained model of face recognition in keras with file. Model can be downloaded from below link: 
# https://www.kaggle.com/suicaokhoailang/facenet-keras#facenet_keras.h5

facenet_model = load_model('/Users/apple/Desktop/facenet_keras.h5')

# Get facial vectors from a detected face

def get_embedding(model, face):
    # scale pixel values
    face = face.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(sample)
    return yhat[0]

# Filepath of image from which face to be detected
all_detected_faces = extract_face('/Users/apple/Desktop/diverse-adults-585x301.jpg')

# Get all facial encoding from the image

all_yemb = []

for faces_arr in all_detected_faces:
    yemb = get_embedding(model=facenet_model, face=faces_arr)
    all_yemb.append(yemb)

# Plotting all the faces that are captured by Facenet and encoding is stored

for each_face in all_detected_faces:
    plt.imshow(each_face)
    plt.show()

# Array of all the faces 128-bit vector encoding 
print(all_yemb)

