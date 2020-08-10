#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:27:42 2020

@author: twarit
"""

# face verification with the VGGFace2 model
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from imutils import paths
import os
import numpy as np
import pickle
import cv2
import imutils
import shutil


# grab the paths to the input images in our dataset
imagePaths = list(paths.list_images('/Users/twarit/Documents/Big Data and AWS/MLProjects/Deep learning solution for Pratwanshi/Example copy'))

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open('/Users/twarit/Documents/Big Data and AWS/MLProjects/Deep learning solution for Pratwanshi/embedding', "rb").read())

print("[INFO] loading keras face recognizer pretrained model convolution layer...")
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

detector = MTCNN()



def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = cv2.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
#	print('print-1',results)
	# extract the bounding box from the first face
	if results!=[]:
		x1, y1, width, height = results[0]['box']
		x2, y2 = x1 + width, y1 + height
	# extract the face
		face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
		if face.shape[0]!= 0 and face.shape[1]!= 0 and face.shape[2]!= 0 :
			image = Image.fromarray(face)
			image=image.resize(required_size)
			face_array = asarray(image)
#		print('print-2',face_array)
			return face_array
		else:
			return []
	else:
		return []

def is_match(known_embedding, candidate_embedding, thresh=0.5):
	# calculate distance between embeddings
	score = cosine(known_embedding, candidate_embedding)
	return score

faces=[]

for (image) in (imagePaths):
	print(image)
	image1 = cv2.imread(image)
    # extract faces
	faces = extract_face(image)
#	print('print-3',faces)
	if faces != []:
		samples = asarray(faces, 'float32')
		samples = preprocess_input(samples, version=2)
		samples=np.expand_dims(samples,axis=0)
		CandidateEmbeddings=model.predict(samples)
		sc=[]
		for iii in range(0,len(data['embeddings'])):
			sc.append(is_match(data['embeddings'][iii], CandidateEmbeddings))
#		print(sc)
		if np.min(sc) < 0.5:
			label=data['names'][np.argmin(sc)]
			shutil.move(image,'/Users/twarit/Documents/Big Data and AWS/MLProjects/Deep learning solution for Pratwanshi/image')
		else:
			label="unknown"
			shutil.move(image,'/Users/twarit/Documents/Big Data and AWS/MLProjects/Deep learning solution for Pratwanshi/Delete this')
		output = imutils.resize(image1, width=400)
    #cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
# show the output image
		print("[INFO] {}".format(label))
	else:
		shutil.move(image,'/Users/twarit/Documents/Big Data and AWS/MLProjects/Deep learning solution for Pratwanshi/Delete this')
    #cv2.imshow("Output", output)
	cv2.waitKey(0)
