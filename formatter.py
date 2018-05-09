#!/usr/bin/env python
import os
import numpy as np
from PIL import Image
import sys
import random
def shuffle(arr):
    for n in range(len(arr) - 1):
        rnd = random.randint(0, (len(arr) - 1))
        val1 = arr[rnd]
        val2 = arr[rnd - 1]

        arr[rnd - 1] = val1
        arr[rnd] = val2

    return arr
def main(orig, dest):
	faces = []
	i = 0
	for file in (y for y in shuffle(os.listdir(orig)) if '.png' in y):
		img = Image.open(orig + '/' + file)
		img_var = img
		#img_var.thumbnail((100, 100))
		img_var.save(dest + '/' + str(i).zfill(3) + '.png', 'PNG')
		faces.append(np.asarray(img_var))
		i += 1
	faces = np.array(faces)
	train = faces[:2000]
	test = faces[2000:]
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
