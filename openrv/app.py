import cv2
import time
from openrv.image import Image, np


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
		self.trackers = {}
		self.capture = cv2.VideoCapture(src) if src != -1 else None
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


	def create_tracker(self, name, tracker_type='csrt'):

		OPENCV_OBJECT_TRACKERS = {
			"csrt": cv2.TrackerCSRT_create,
			"kcf": cv2.TrackerKCF_create,
			"boosting": cv2.TrackerBoosting_create,
			"mil": cv2.TrackerMIL_create,
			"tld": cv2.TrackerTLD_create,
			"medianflow": cv2.TrackerMedianFlow_create,
			"mosse": cv2.TrackerMOSSE_create
		}
		self.trackers[name] = OPENCV_OBJECT_TRACKERS[tracker_type]()


	def set_tracker(self, name, frame, roi):
		frame = frame if type(frame) == np.ndarray else frame.img
		self.trackers[name].init(frame, roi)

	def get_tracker(self, name, image):
		frame = image if type(image) == np.ndarray else image.img
		success, box = self.trackers[name].update(frame)
		box = tuple([int(v) for v in box])
		return success, (box[:2], box[2:])

	def add_trackbar(self, window, name, minimum, maximum, action=lambda a: None):
		cv2.namedWindow(window)
		cv2.createTrackbar(name, window, minimum, maximum, action)

	def get_trackbar(self, window, name):
		return cv2.getTrackbarPos(name, window)
