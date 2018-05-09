from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np
import sys

def rescale_image(image, orig_minmax, dest_minmax):
	orig_range = orig_minmax[1] - orig_minmax[0]
	dest_range = dest_minmax[1] - dest_minmax[0]

	image = (image - orig_minmax[0]) / float(orig_range)
	return image * dest_range + dest_minmax[0]

class ConvAAE():
	def __init__(self, train_loc):
		self.serial = 'conv_aae_gloss99_'
		self.train_loc = train_loc

		img_rows = 256
		img_cols = 256
		channels = 3
		self.img_shape = (img_rows, img_cols, channels)
		self.encoded_dim = 16

		optimizer = Adam(0.0002, 0.5)

		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy',
			optimizer=optimizer,
			metrics=['accuracy'])

		self.encoder = self.build_encoder()
		print self.encoder.output_shape
		self.decoder = self.build_decoder()

		img = Input(shape=self.img_shape)
		enc = self.encoder(img)
		gen_img = self.decoder(enc)

		self.discriminator.trainable = False

		validity = self.discriminator(enc)

		gan_loss_weight = 0.99

		self.convolutional_aae = Model(img, [gen_img, validity])
		self.convolutional_aae.compile(loss=['mse', 'binary_crossentropy'],
			loss_weights=[gan_loss_weight, 1 - gan_loss_weight],
			optimizer=optimizer)

	def build_encoder(self):
		encoder = Sequential()
		encoder.add(Conv2D(32, (5, 5), activation='relu', input_shape=self.img_shape))
		encoder.add(MaxPooling2D())
		encoder.add(Dropout(0.1))
		encoder.add(Conv2D(32, (4, 4), strides=2, activation='relu'))
		encoder.add(MaxPooling2D())
		encoder.add(BatchNormalization(momentum=0.8))
		encoder.add(Dropout(0.1))
		encoder.add(Flatten())
		encoder.add(Dense(256))
		encoder.add(LeakyReLU(alpha=0.2))
		encoder.add(BatchNormalization(momentum=0.8))
		encoder.add(Dropout(0.2))
		encoder.add(Dense(self.encoded_dim))

		img = Input(shape=self.img_shape)
		enc = encoder(img)

		encoder.summary()
		return Model(img, enc)

	def build_decoder(self):
		decoder = Sequential()
		# Upsample for the deconvolution
		decoder.add(Dense(self.encoded_dim, input_dim=self.encoded_dim))
		decoder.add(LeakyReLU(alpha=0.2))
		decoder.add(Dense(32 * 32 * 32))
		decoder.add(LeakyReLU(alpha=0.2))
		decoder.add(Dropout(0.2))
		decoder.add(Reshape((32, 32, 32)))
		# The following layers were removed to improve performance - no noticable drop in quality
		# Deconvolution
		#
		# Upsample
		#decoder.add(Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same'))
		#decoder.add(Dropout(0.1))
		#decoder.add(Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same'))
		#decoder.add(Dropout(0.1))
		decoder.add(Conv2DTranspose(16, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same'))
		decoder.add(Dropout(0.1))
		decoder.add(Conv2DTranspose(16, kernel_size=(8, 8), strides=(2, 2), activation='relu', padding='same'))
		decoder.add(Dropout(0.1))
		decoder.add(Conv2DTranspose(32, kernel_size=(8, 8), strides=(2, 2), activation='relu', padding='valid'))
		decoder.add(Dropout(0.1))
		# Squash image to right size
		decoder.add(Conv2D(self.img_shape[-1], kernel_size=7, activation='tanh', padding='valid'))

		enc = Input(shape=(self.encoded_dim, ))
		gen_img = decoder(enc)

		decoder.summary()
		return Model(enc, gen_img)

	def build_discriminator(self):
		# The discriminator classifies the encoded layer

		discriminator = Sequential()
		discriminator.add(Dense(256, input_dim=self.encoded_dim))
		discriminator.add(LeakyReLU(alpha=0.2))
		discriminator.add(Dropout(0.1))
		discriminator.add(Dense(128, input_dim=self.encoded_dim))
		discriminator.add(LeakyReLU(alpha=0.2))
		discriminator.add(Dropout(0.1))
		discriminator.add(Dense(1, activation='sigmoid'))

		enc = Input(shape=(self.encoded_dim, ))
		validity = discriminator(enc)

		discriminator.summary()

		return Model(enc, validity)

	def train(self, epochs, batch_size):

		train_imgs = np.load(self.train_loc + '/all_faces.fc')
		train_imgs = rescale_image(train_imgs, (0, 255), (-1, 1))

		for epoch in range(epochs):

			# Discriminator Training
			half_batch_size = batch_size / 2
			indices = np.random.randint(0, train_imgs.shape[0], half_batch_size)
			half_batch = train_imgs[indices]

			valid = np.ones((half_batch_size, 1))
			invalid = np.zeros((half_batch_size, 1))

			real_encs = self.encoder.predict(half_batch)
			rand_encs = np.random.normal(size=(half_batch_size, self.encoded_dim))

			d_loss = (self.discriminator.train_on_batch(rand_encs, valid), 
						self.discriminator.train_on_batch(real_encs, invalid))

			d_loss = [sum(l) / 2.0 for l in zip(*d_loss)] # Gross, I know

			# Generator Training
			valid = np.ones((batch_size, 1))
			batch = train_imgs[np.random.randint(0, train_imgs.shape[0], batch_size)]

			g_loss = self.convolutional_aae.train_on_batch(batch, [batch, valid])

			print "Epoch", epoch
			print "Generator loss, mse -----", g_loss[0], g_loss[1], "Discriminator loss, acc -", d_loss[0], d_loss[1] * 100

			if not epoch % 50:
				self.save_images(epoch, train_imgs[np.random.randint(0, train_imgs.shape[0], 25)])
				self.save_model()
				print '--Saved--'
		encodings = self.encoder.predict(train_imgs)

	def save_images(self, epoch, imgs):
		r, c = 5, 5

		encoded_imgs = self.encoder.predict(imgs)
		mean_encoding = np.mean(encoded_imgs, axis=0).reshape(1, self.encoded_dim)
		gen_imgs = self.decoder.predict(encoded_imgs)
		avg_img = self.decoder.predict(mean_encoding)

		gen_imgs = rescale_image(gen_imgs, (-1, 1), (0, 1))
		avg_img = rescale_image(avg_img, (-1, 1), (0, 1))

		fig, axs = plt.subplots(r + 1, c)
		cnt = 0

		encs_from_rand = np.random.normal(0, 4, size=(c, self.encoded_dim))
		imgs_from_rand = self.decoder.predict(encs_from_rand)
		imgs_from_rand = rescale_image(imgs_from_rand, (-1, 1), (0, 1))

		for i in range(r + 1):
			for j in range(c):
				if i == r:
					axs[i,j].imshow(imgs_from_rand[j])
					axs[i,j].axis('off')
				elif i == j == 2:
					axs[i,j].imshow(avg_img[0])
					axs[i,j].axis('off')
				else:
					axs[i,j].imshow(gen_imgs[cnt])
					axs[i,j].axis('off')
				cnt += 1
		fig.savefig("conv_aae/conv_aae_gloss99_%d.png" % epoch,dpi=300)
		plt.close()

	def save_model(self):

		def save(model, model_name):
			model_path = "conv_aae/%s.json" % model_name
			weights_path = "conv_aae/%s_weights.hdf5" % model_name
			options = {"file_arch": model_path,
						"file_weight": weights_path}
			json_string = model.to_json()
			open(options['file_arch'], 'w').write(json_string)
			model.save_weights(options['file_weight'])

		save(self.encoder, self.serial + "-aae_encoder")
		save(self.decoder, self.serial + "-aae_decoder")
		save(self.discriminator, self.serial + "-aae_discriminator")

if __name__ == '__main__':
	if len(sys.argv) != 2:
		raise ValueError('Usage: python conv_aae.py <training image location>')
	train_loc = sys.argv[1]
	aae = ConvAAE(train_loc)
	aae.train(10001, 32)
