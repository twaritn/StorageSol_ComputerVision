# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# face verification with the VGGFace2 model
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import os
import numpy as np
import pickle
from imutils import paths

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images('/Users/twarit/Documents/Big Data and AWS/MLProjects/Deep learning solution for Pratwanshi/Dataset'))

print("[INFO] loading keras face recognizer pretrained model convolution layer...")
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

print("[INFO] loading keras multi tasking face detection convolution layer...")
detector = MTCNN()


knownEmbeddings = []
knownNames = []

# initialize the total number of faces processed
total = 0

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# extract faces and calculate face embeddings for a list of photo files
for (image) in (imagePaths):
    # extract faces
    faces = extract_face(image)
    samples = asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    samples=np.expand_dims(samples,axis=0)
    knownEmbeddings.append(model.predict(samples))
    label = image.split(os.path.sep)[-2]
    knownNames.append(label)
    total=total+1

# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open( '/Users/twarit/Documents/Big Data and AWS/MLProjects/Deep learning solution for Pratwanshi/embedding', "wb")
f.write(pickle.dumps(data))
f.close()