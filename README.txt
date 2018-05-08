Convolutional Autoencoder for Faces.
Includes program to recreate faces from 16-dimensional encoding, visually represent modifications to encoding, and transition smoothly between faces.

Usage:
If using a new dataset, first run formatter.py on a folder of images to reduce to 100x100 and create numpy dump of image data.
Then train with conv_aae, which should save an encodings numpy dump as well as the model. (Requires h5py installation)
Training should take an hour or two if training on a laptop CPU.
Then run slider.py to pull up representations of chosen faces and transition between them.

If using a pre-trained model, be sure to put the model inside FaceVectors/conv_aae and the encoding dump inside FaceVectors/

Nueva Students: don't share the facebook dataset outside the school.
Share and Enjoy.