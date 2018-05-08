#!/usr/bin/env python
import os
import numpy as np
from PIL import Image
import sys

def main(orig, dest):
	faces = []
	i = 0
	for file in (y for y in os.listdir(orig) if '.jpg' in y):
		img = Image.open(orig + '/' + file)
		img_var = img
		img_var.thumbnail((100, 100))
		img_var.save(dest + '/' + str(i).zfill(3) + '.jpg', 'JPEG')
		faces.append(np.asarray(img_var))
		i += 1
	faces = np.array(faces)
	train = faces[:320]
	test = faces[320:]
	faces.dump(dest + '/all_faces.fc')
	#train.dump('Nuevaphotos/Formatted5/train_faces.fc')
	#test.dump('Nuevaphotos/Formatted5/test_faces.fc')
	print faces.shape
	print train.shape
	print test.shape	


if __name__ == '__main__':
	if len(sys.argv) != 3:
		raise ValueError('Usage: python formatter.py <image folder> <destination folder>')
	face_folder = sys.argv[1]
	dest_folder = sys.argv[2]
	main(face_folder, dest_folder)