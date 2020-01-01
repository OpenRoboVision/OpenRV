import cv2
import time
from openrv.image import Image


class CaptureException(Exception):
	def __init__(self, msg):
		Exception.__init__(self, "Something went wrong: {}".format(msg))


class AccessException(Exception):
	def __init__(self, msg):
		Exception.__init__(self, "Can't get an access to component: \"{}\". Maybe it wasn't defined".format(msg))


class App:

	def __init__(self, src=-1, writer=None, quit_key='q'):
		self.run = True
		self.fps = 0
		self.quit_key = quit_key
		self.writer = None
		self.writer_data = None
		self.capture = cv2.VideoCapture(src)
		if src != -1 and (self.capture is None or not self.capture.isOpened()):
			raise CaptureException("Can't open capture with src = {}".format(src))
		if not (writer is None):
			self.writer_data = writer
			self.writer = cv2.VideoWriter(writer['fname'],
										  cv2.VideoWriter_fourcc(*writer['codec']),
										  float(writer['fps']),
										  tuple(writer['shape']))

	def start(self, body):
		while self.run:
			start_time = time.time()
			key = chr(cv2.waitKey(1) & 0xFF)
			data = {'key': key, 'fps': self.fps, 'writer': None, 'frame': None, 'self': self}
			if self.capture is not None:
				data['frame'] = Image.from_arr(self.capture.read()[1])
			if self.writer is not None:
				data['writer'] = [self.writer, self.writer_data]

			last_resp = None
			for func in body:
				resp = func(last_resp, data, self)
				last_resp = resp

			self.fps = 1.0 / (time.time() - start_time)
			if key == self.quit_key:
				self.finish()

	def finish(self):
		self.run = False
		if self.writer is not None:
			self.writer.release()
		if self.capture is not None:
			self.capture.release()
		cv2.destroyAllWindows()


	def add_trackbar(self, window, name, minimum, maximum, action=lambda a: None):
		cv2.namedWindow(window)
		cv2.createTrackbar(name, window, minimum, maximum, action)

	def get_trackbar(self, window, name):
		return cv2.getTrackbarPos(name, window)
