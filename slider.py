import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from matplotlib.widgets import Slider, Button

class Decoder():
	def __init__(self, folder, name):
		dec_file = open(folder + name + '-aae_decoder.json', 'r').read()	
		self.model = model_from_json(dec_file)	
		self.model.load_weights(folder + name + '-aae_decoder_weights.hdf5')

		self.encodings = np.load('encoding_' + name + '.en')
		print 'stdev\n', np.std(self.encodings, axis=0)
		print 'range\n', np.ptp(self.encodings, axis=0)
		print 'avg\n', np.mean(self.encodings, axis=0)

		self.enc = None
		self.get_face()
		self.encoded_shape = self.enc.shape[0]
		self.gen_face = self.enc_to_face()

		print self.enc

	def get_face(self):
		face_num = input('Give Face Number\n')
		self.enc = self.encodings[face_num]

	def enc_to_face(self):
		return self.model.predict(self.enc.reshape(1, self.encoded_shape))[0] * 0.5 + 0.5

	def slider_show(self):
		fig, ax = plt.subplots(figsize=(8, 7))
		plt.subplots_adjust(left=0.25, bottom=0.5)
		l = plt.imshow(self.gen_face)

		axcolor = 'lightgoldenrodyellow'

		n_axes = []
		n_sliders = []
		for i in range(self.encoded_shape):
			n_axes.append(plt.axes([0.25, 0.4 - 0.025 * i, 0.65, 0.01], facecolor=axcolor))
			n_sliders.append(Slider(n_axes[i], 'Neuron' + str(i), -10, 10, valinit=self.enc[i]))

		def update(val):
			for i, sld in enumerate(n_sliders):
				self.enc[i] = sld.val
			l.set_data(self.enc_to_face())
			fig.canvas.draw_idle()
		for sld in n_sliders:
			sld.on_changed(update)

		button = Button(plt.axes([0.52, 0.9, 0.1, 0.04]), 'Reset', color=axcolor, hovercolor='0.975')

		def reset(event):
			for sld in n_sliders:
				sld.reset()
		button.on_clicked(reset)

		plt.show()
		plt.close()

	def slider_transition(self):
		self.get_face()
		orig_enc = self.enc
		self.get_face()
		end_enc = self.enc

		fig, ax = plt.subplots(figsize=(8, 7))
		plt.subplots_adjust(left=0.25, bottom=0.5)
		l = plt.imshow(self.gen_face)

		axcolor = 'lightgoldenrodyellow'

		t_axis = plt.axes([0.25, 0.4, 0.65, 0.01], facecolor=axcolor)
		t_slider = Slider(t_axis, 'Transition', 0, 1, valinit=0)

		def update(val):
			t = t_slider.val
			self.enc = end_enc * t + orig_enc * (1 - t)
			l.set_data(self.enc_to_face())
			fig.canvas.draw_idle()
		t_slider.on_changed(update)

		button = Button(plt.axes([0.52, 0.9, 0.1, 0.04]), 'Reset', color=axcolor, hovercolor='0.975')

		def reset(event):
			for sld in n_sliders:
				sld.reset()
		button.on_clicked(reset)

		plt.show()
		plt.close()


dec = Decoder('conv_aae/', 'conv_aae_gloss99_')
dec.slider_show()
dec.slider_transition()