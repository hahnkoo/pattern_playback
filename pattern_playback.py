"""A Digital Pattern Playback System in Python"""

__author__ = "Hahn Koo (hahn.koo@sjsu.edu)"
__date__ = "May 19, 2022"

import argparse, sys, tkinter
import numpy as np
from PIL import Image, ImageGrab, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy import signal
from scipy.io import wavfile


### Functions for image processing

def get_x_and_y(event):
	"""Get (x, y) coordinate of cursor on canvas."""
	global last_x, last_y
	last_x, last_y = event.x, event.y

def draw(event):
	"""Draw on canvas."""
	global last_x, last_y, brush_fill, brush_width, canvas, image_draw
	canvas.create_line((last_x, last_y, event.x, event.y), fill=brush_fill, width=brush_width)
	image_draw.line((last_x, last_y, event.x, event.y), fill=brush_fill, width=brush_width)
	last_x, last_y = event.x, event.y

def erase(event):
	"""Erase (draw in white) on canvas."""
	global last_x, last_y, erase_fill, erase_width, canvas, image_draw
	canvas.create_line((last_x, last_y, event.x, event.y), fill=erase_fill, width=erase_width)
	image_draw.line((last_x, last_y, event.x, event.y), fill=erase_fill, width=erase_width)
	last_x, last_y = event.x, event.y

def add_gridlines_to_canvas(width, height, margin, sampling_rate, duration):
	"""Add grid lines to canvas."""
	global canvas
	n_gridlines = 20
	font_size = min(margin // 4, 6)
	x_step = (width - 2 * margin) // n_gridlines
	y_step = (height - 2 * margin) // n_gridlines
	time = duration / n_gridlines
	time_precision = len(str(time).split('.')[-1])
	for line in range(margin + x_step, width - margin, x_step):
		canvas.create_line([(line, margin), (line, height - margin)], fill='blue', dash=2)
		canvas.create_text(line, height - (margin // 2), text=str(round(time, time_precision)), font="Arial " + str(font_size))
		time += duration / n_gridlines
	frequency = (sampling_rate / 2)  - sampling_rate / (2 * n_gridlines)
	for line in range(margin + y_step, height - margin, y_step):
		canvas.create_line([(margin, line), (width - margin, line)], fill='blue', dash=2)
		canvas.create_text(margin // 2, line, text=str(round(frequency)), font="Arial " +str(font_size))
		frequency -= sampling_rate / (2 * n_gridlines)

def remove_top_gridline(image, threshold=10):
	"""Remove top grid line, if present, from the image."""
	center_25 = image.shape[1] // 4
	center_75 = center_25 * 3
	center_half = image[:, center_25:center_75]
	top_50 = image.shape[0] // 2
	means = np.mean(center_half[:top_50] , axis=1)
	diff = means[:-1] - means[1:]
	ts = 0
	diff_threshold = np.mean(diff[diff > 0])
	stop = False
	while not stop:
		end_of_list = (ts >= len(diff) - 1)
		found_match = diff[ts] > diff_threshold and np.std(center_half[ts + 1]) < threshold
		stop = end_of_list or found_match
		ts += 1
	te = 0
	if np.std(center_half[ts]) < threshold:
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


### Spectrogram class

class Spectrogram:

	def __init__(self, image, crop=False, sampling_rate=16000, duration=1):
		self.raw = np.array(image.convert('L'))
		if crop:
			self.fig, self.ax = plt.subplots(2, 1)
			self.ax[0].imshow(self.raw)
			self.area = RectangleSelector(self.ax[0], self.onselect, button=[1])
			plt.show()
			self.cropped = remove_gridlines(self.cropped)
		else: self.cropped = remove_gridlines(self.raw)
		target_length = int(sampling_rate * duration)
		self.image2spectrogram(target_length)

	def onselect(self, eclick, erelease):
		"""Define what happens when selecting a portion."""
		self.cropped = self.raw[int(self.area.extents[2]):int(self.area.extents[3]), int(self.area.extents[0]):int(self.area.extents[1])]
		self.ax[1].imshow(self.cropped)

	def to_array(self):
		"""Convert cropped image to a proper array."""
		self.X = self.cropped[::-1] # (time, frequency) -> (frequency, time)
		self.X = 256 - self.X # high pixel (bright) -> low energy
	
	def interpret_as_dB(self):
		"""Interpret pixel values as dB values."""
		self.X = self.X / 2 # assume 256/2 is the max dB
		self.X = 10 ** (self.X / 20) # assume 1 is reference

	def resize_target(self, target_length):
		"""Calculate how spectrogram must be resized to synthesize waveform of target_length samples."""
		n_rows, n_cols = self.X.shape
		coeff = [n_cols, -(n_cols + n_rows), n_rows * (1 - 2 * target_length)]
		m = int(max(np.roots(coeff)))
		if m % 2 == 0: m += 1 # griffin-lim and istft are not always the same if n_row is even
		n = int(m * n_cols / n_rows)
		return m, n

	def resize(self, target_length):
		"""Resize spectrogram."""
		m, n = self.resize_target(target_length)
		new_x = signal.resample(self.X, m, axis=0)
		new_x = signal.resample(new_x, n, axis=1)
		self.X = new_x

	def image2spectrogram(self, target_length):
		"""Reconstruct spectrogram from image."""
		self.to_array()
		old_shape = self.X.shape
		self.resize(target_length)
		new_shape = self.X.shape
		self.interpret_as_dB()
		sys.stderr.write('# Reconstructed spectrogram from image: ' + str(old_shape) + ' -> ' + str(new_shape) + '\n')

	def to_waveform(self, sampling_rate, griffinlim=False):
		"""Convert to waveform."""
		method = 'zero-phase ISTFT'
		nperseg = (self.X.shape[0] - 1) * 2 # nperseg = n_fft
		noverlap = nperseg // 4 * 3
		hopsize = nperseg - noverlap
		if griffinlim:
			method = 'Griffin-Lim algorithm'
			import librosa
			x = librosa.griffinlim(self.X)
			t = np.arange(len(x)) / sampling_rate
		else:
			t, x = signal.istft(self.X, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
		sys.stderr.write('# Synthesized waveform from spectrogram using ' + method + ':' + '\n - sampling rate = ' + str(args.sampling_rate) + '\n - nperseg = n_fft = ' + str(nperseg) + '\n - hopsize = ' + str(hopsize) +'\n - resulting duration = ' + str(t[-1]) + '\n')
		return t, x

	def plot(self, ax, sampling_rate=None):
		"""Plot magnitude spectrogram."""
		x, y = np.arange(self.X.shape[1]), np.arange(self.X.shape[0])
		if sampling_rate:
			x = ((len(y) - 1) // 2) / sampling_rate * x
			y = sampling_rate / (2 * len(y)) * y
		ax.pcolormesh(x, y, 20 * np.log(np.abs(self.X)), cmap='Greys', shading='auto')
		ax.set_ylim(top=sampling_rate / 2, bottom=0)
		ax.set_xlabel('time (seconds)')



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--duration', type=float, default=1, required=True)
	parser.add_argument('--sampling_rate', type=int, default=16000, required=True)
	parser.add_argument('--load', help='img file to load')
	parser.add_argument('--crop', action='store_true', help='whether to crop from image')
	parser.add_argument('--draw', action='store_true', help='whether to draw a spectrogram on a blank canvas')
	parser.add_argument('--canvas_width', type=int, default=1200, help='canvas width')
	parser.add_argument('--canvas_height', type=int, default=600, help='canvas width')
	parser.add_argument('--canvas_margin', type=int, default=50, help='canvas margin outside the grid')
	parser.add_argument('--save_drawing', default='spectrogram.png', help='save the spectrogram you drew on canvas as')
	parser.add_argument('--show_graphs', action='store_true', help='whether to display the loaded spectrogram, reconstructed waveform and its spectrogram')
	parser.add_argument('--griffinlim', action='store_true', help='whether to use the Griffin-Lim algorithm to reconstruct the waveform instead of assuming zero phase')
	parser.add_argument('--save_wav', default='out.wav', help='save the reconstructed waveform as')
	args = parser.parse_args()

	if not args.load is None:
		image = Image.open(args.load)
	elif args.draw:
		last_x = 0; last_y = 0
		brush_fill = 'black'; brush_width = 5
		erase_fill = 'white'; erase_width = 5
		root = tkinter.Tk()
		canvas = tkinter.Canvas(root, height=args.canvas_height, width=args.canvas_width, background='white')
		image = Image.new('RGB', (args.canvas_width, args.canvas_height), 'white')
		image_draw = ImageDraw.Draw(image)
		canvas.create_rectangle(args.canvas_margin, args.canvas_margin, args.canvas_width - args.canvas_margin, args.canvas_height - args.canvas_margin, outline='black')
		image_draw.rectangle([args.canvas_margin, args.canvas_margin, args.canvas_width - args.canvas_margin, args.canvas_height - args.canvas_margin], outline='black')
		canvas.bind('<Button-1>', get_x_and_y)
		canvas.bind('<B1-Motion>', draw)
		canvas.bind('<Button-3>', get_x_and_y)
		canvas.bind('<B3-Motion>', erase)
		add_gridlines_to_canvas(args.canvas_width, args.canvas_height, args.canvas_margin, args.sampling_rate, args.duration)
		canvas.pack()
		root.mainloop()
		image.save(args.save_drawing)
	s = Spectrogram(image, crop=args.crop, sampling_rate=args.sampling_rate, duration=args.duration)
	t, x = s.to_waveform(args.sampling_rate, griffinlim=args.griffinlim)
	if args.show_graphs:
		fig, ax = plt.subplots(2, 1)
		s.plot(ax[0], sampling_rate=args.sampling_rate)
		ax[0].set_title('Spectrogram reconstructed from image')
		ax[1].plot(t, x, color='black')
		ax[1].set_title('Synthesized waveform')
		ax[1].set_xlim(left=0, right=t[-1])
		fig.subplots_adjust(hspace=0.5)
		plt.show()
	wavfile.write(args.save_wav, args.sampling_rate, x)
	sys.stderr.write('# Saved waveform as ' + args.save_wav + '\n')