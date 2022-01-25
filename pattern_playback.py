"""Pattern Playback"""

__author__ = "Hahn Koo (hahn.koo@sjsu.edu)"
__date__ = "January 24, 2022"

import argparse
import numpy as np
import tkinter
from PIL import Image, ImageGrab, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy import signal
from scipy.io import wavfile

def get_x_and_y(event):
	global last_x, last_y
	last_x, last_y = event.x, event.y

def draw(event):
	global last_x, last_y, brush_fill, brush_width, canvas, image_draw
	canvas.create_line((last_x, last_y, event.x, event.y), fill=brush_fill, width=brush_width)
	image_draw.line((last_x, last_y, event.x, event.y), fill=brush_fill, width=brush_width)
	last_x, last_y = event.x, event.y

def remove_top_gridline(image):
	"""Remove top grid line from the image."""
	center_25 = image.shape[1] // 4
	center_75 = center_25 * 3
	center_half = image[:, center_25:center_75]
	top_50 = image.shape[0] // 2
	means = np.mean(center_half[:top_50] , axis=1)
	diff = means[:-1] - means[1:]
	ts = np.argmax(diff) + 1
	te = ts
	while means[ts] == means[te]: te += 1
	return image[te:]

def remove_gridlines(image):
	"""Remove gridlines -- top, bottom, left, right -- from image."""
	image = remove_top_gridline(image)
	image = remove_top_gridline(image[::-1])[::-1]
	image = remove_top_gridline(image.T).T
	image = remove_top_gridline(image.T[::-1])[::-1].T
	return image

def conv1d(x, size, stride, pad, window, axis=0):
	"""Apply 1D convolution to filter x."""
	if axis != 0: x = x.T
	n = (len(x) - size + 2 * pad) // stride + 1
	x = np.pad(x, ((pad, pad), (0, 0)))
	w = signal.windows.get_window(window, size)
	y = [np.dot(w, x[i * stride : i * stride + size, :]) for i in range(n)]
	y = np.array(y)
	if axis != 0: y = y.T
	return y


class Spectrogram:

	def __init__(self, image, crop=False, dB=False):
		self.raw = np.array(image.convert('L'))
		if crop:
			self.fig, self.ax = plt.subplots(2, 1)
			self.ax[0].imshow(self.raw)
			self.area = RectangleSelector(self.ax[0], self.onselect, drawtype='box', button=[1])
			plt.show()
			self.cropped = remove_gridlines(self.cropped)
		else: self.cropped = remove_gridlines(self.raw)
		self.to_array(dB)		

	def onselect(self, eclick, erelease):
		"""Define what happens when selecting a portion."""
		self.cropped = self.raw[int(self.area.extents[2]):int(self.area.extents[3]), int(self.area.extents[0]):int(self.area.extents[1])]
		self.ax[1].imshow(self.cropped)


	def to_array(self, dB):
		"""Convert cropped image to a frequency x time array."""
		self.X = self.cropped[::-1] # frequency by time
		self.X = 255 - self.X
		self.X = self.X / 100
		if dB: self.X = 10 ** (self.X / 20)

	def to_waveform(self, sampling_rate):
		"""Convert to waveform."""
		t, x = signal.istft(self.X, fs=sampling_rate)
		return t, x

	def restructure(self, n_total_samples, n_frequency_bins=256, window='boxcar'):
		n_frames = n_total_samples // n_frequency_bins
		# frequency axis
		stride = self.X.shape[0] // (n_frequency_bins + 1)
		size = 2 * stride
		self.X = conv1d(self.X, size, stride, 0, window=window, axis=0)
		# time axis
		stride = self.X.shape[1] // (n_frames + 1)
		size = 2 * stride
		self.X = conv1d(self.X, size, stride, 0, 'boxcar', axis=1)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--duration', type=float, default=1, required=True)
	parser.add_argument('--sampling_rate', type=int, default=16000, required=True)
	parser.add_argument('--save_wav', default='out.wav', required=True)
	parser.add_argument('--save_spectrogram', default='spectrogram.png')
	parser.add_argument('--load')
	parser.add_argument('--num_frequency_bins', type=int, default=128)
	parser.add_argument('--window', default='boxcar', help='type of window for smoothing pixels')
	parser.add_argument('--show_graphs', action='store_true')
	args = parser.parse_args()

	N = int(args.duration * args.sampling_rate)
	if args.load is None:
		last_x = 0; last_y = 0
		brush_fill = 'black'; brush_width = 5
		root = tkinter.Tk()
		canvas = tkinter.Canvas(root, height=450, width=850, background='white')
		image = Image.new('RGB', (850, 450), 'white')
		image_draw = ImageDraw.Draw(image)
		canvas.create_rectangle(25, 25, 825, 425, outline='black')
		image_draw.rectangle([25, 25, 825, 425], outline='black')
		canvas.bind('<Button-1>', get_x_and_y)
		canvas.bind('<B1-Motion>', draw)
		canvas.pack()
		root.mainloop()
		image.save(args.save_spectrogram)
	else:
		image = Image.open(args.load)
	s = Spectrogram(image, dB=True)
	s.restructure(N, n_frequency_bins=args.num_frequency_bins)
	t, x = s.to_waveform(args.sampling_rate)
	if args.show_graphs:
		fig, ax = plt.subplots(2, 1)
		ax[0].pcolormesh(20 * np.log(s.X))
		ax[1].plot(t, x)
		plt.show()
	wavfile.write(args.save_wav, args.sampling_rate, x)