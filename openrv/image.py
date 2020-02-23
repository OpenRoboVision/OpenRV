import numpy as np
import cv2
import imutils
import cv2.aruco as aruco
from openrv.colors import *
from openrv.contour import Contour, ContourSet
import skimage.filters
import skimage.morphology
import skimage.feature
from scipy import ndimage as ndi
from imutils.perspective import order_points
from openrv.defines import BGR, GRAY, HSV

aruco_params = aruco.DetectorParameters_create()

class Image:

	def __init__(self):
		self.image = np.array([])
		self.color_scheme = BGR

	def show(self, name='Frame', freeze=False, id=None):
		""" 
		Show self.image

		Will draw this image on the window. Will create it if doesn't exists

		Parameters: 
			name : Name of the window to draw on

		Returns: 
			Image: self 

		"""
		if id is not None:
			name += ' ' + str(id)
		cv2.imshow(name, self.image)
		if freeze:
			cv2.waitKey()
		return self

	def resize(self, width=None, height=None, inter=cv2.INTER_CUBIC):
		""" 
		Resize self.image with given params

		Parameters: 
			width  : Target width
			height : Target height
			inter  : interpolation type

		Returns: 
			self: Image

		"""
		if width is None or height is None:
			self.image = imutils.resize(self.image, width=width, height=height, inter=inter)
		else:
			self.image = cv2.resize(self.image, tuple(map(int, (width, height))))
		return self

	def gray(self):
		""" 
		Creates a grayscale copy of the image

		Returns: 
			Image: GRAYSCALE copy

		"""
		if self.color_scheme == GRAY:
			return self.copy()
		res = self.copy()
		res.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		res.color_scheme = GRAY
		return res

	def to_gray(self):
		""" 
		Convert this image to GRAYSCALE

		Returns: 
			Image: self

		"""
		if self.color_scheme == GRAY:
			return self
		self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		self.color_scheme = GRAY
		return self

	def bgr(self):
		""" 
		Creates a BGR copy of the image

		Returns: 
			Image: BGR copy

		"""
		res = self.copy()
		if self.color_scheme == HSV:
			res.image = cv2.cvtColor(self.image, cv2.COLOR_HSV2BGR)
			res.color_scheme = BGR
		elif self.color_scheme == GRAY:
			res.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
			res.color_scheme = BGR
		return res

	def to_bgr(self):
		""" 
		Convert this image to BGR

		Returns: 
			Image: self

		"""
		if self.color_scheme == HSV:
			self.image = cv2.cvtColor(self.image, cv2.COLOR_HSV2BGR)
			self.color_scheme = BGR
		elif self.color_scheme == GRAY:
			self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
			self.color_scheme = BGR
		return self

	def hsv(self):
		""" 
		Creates a HSV copy of the image

		Returns: 
			Image: grayscaled copy

		"""
		res = self.copy()
		if self.color_scheme == BGR:
			res.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
			res.color_scheme = HSV
		elif self.color_scheme == GRAY:
			res.image = cv2.cvtColor(cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
			res.color_scheme = HSV
		return res

	def to_hsv(self):
		""" 
		Convert this image to HSV

		Returns: 
			Image: self

		"""
		if self.color_scheme == BGR:
			self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
			self.color_scheme = HSV
		elif self.color_scheme == GRAY:
			self.image = cv2.cvtColor(cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
			self.color_scheme = HSV
		return self

	def rotate(self, angle, bound=False):
		""" 
		Rotate self.image on given angle

		Parameters: 
			angle (int)  : rotatation angle (CCW)
			bound (bool) : crop the image or not

		Returns: 
			self: Image

		"""
		if bound:
			self.image = imutils.rotate_bound(self.image, angle)
		else:
			self.image = imutils.rotate(self.image, -angle)
		return self

	def crop(self, x1, y1, x2, y2):
		""" 
		Crop a part of self.image
		"""
		if x2 is None: x2 = self.width
		if y2 is None: y2 = self.height
		if x2 < 0: x2 = self.width + x2
		if y2 < 0: y2 = self.height + y2
		self.image = self.image[int(y1):int(y2), int(x1):int(x2)]
		return self

	def scale(self, val):
		self.resize(width=self.width * val)

	def apply_mask(self, mask):
		""" 
		Apply the bitwise mask to self.image

		Parameters: 
			mask (Image) : bitwise mask to apply

		Returns: 
			self: Image

		"""
		mask = mask.copy().gray()
		self.image = cv2.bitwise_and(self.image, self.image, mask=mask.image)
		return self

	def find_color(self, base_color=None, sensitivity=25, lower=(0, 0, 0), upper=(255, 255, 255), sens_mult=1.0):
		""" 
		Search for the given color on self.image

		self.image will be a bitwise mask of the color

		Parameters: 
			base_color (tuple) : color to find (if != None)
			sensitivity (int)  : sensativity for base_color (if base_color != None)
			sens_mult (float)  : sensativity factor (if base_color != None)
			lower (tuple)      : lower limit of the color range (if base_color == None)
			upper (tuple)      : upper limit of the color range (if base_color == None)

		Returns: 
			self: Image

		"""
		res = self.copy()
		if res.color_scheme != HSV:
			res.to_hsv()

		if base_color is not None:
			lower = [base_color[0] - sensitivity, base_color[1] - (sens_mult * sensitivity),
					 base_color[2] - (sens_mult * sensitivity)]
			upper = [base_color[0] + sensitivity, base_color[1] + (sens_mult * sensitivity),
					 base_color[2] + (sens_mult * sensitivity)]

		self.image = cv2.inRange(res.img, np.array(lower), np.array(upper))
		return self

	def find_aruco(self, dict_type=aruco.DICT_ARUCO_ORIGINAL):
		""" 
		Returns list of corners and ids of found markers

		Parameters: 
			dict_type (int) : ArUco dictionary type

		Returns: 
			corners: list[list]

		"""
		corners, ids, _ = aruco.detectMarkers(self.image, aruco.Dictionary_get(dict_type),
											  parameters=aruco_params)
		return corners, ids

	def draw_aruco(self, corners, ids):
		""" 
		Draw ArUco markers on self.image
		"""
		cv2.aruco.drawDetectedMarkers(self.image, corners, ids)
		return self

	def correct_perspective(self, src, dst):
		""" 
		Remove perspective distortion (4 points)

		Parameters: 
			src (list) : Four points(x,y) on the original image
			dst (list) : 4 target points after correction

		Returns: 
			self: Image

		"""
		img = self.image
		h, w = img.shape[:2]
		M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
		self.image = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
		return self

	def wrap_perspective_rect(self, points, width, height):
		if len(points) != 4:
			raise ValueError('Contour\'s number of curners must be equal to 4')

		src = order_points(points.reshape(4, 2))
		dst = np.array([
			[0, 0],
			[width, 0],
			[width, height],
			[0, height]
		])
		self.correct_perspective(src, dst).crop(0, 0, width, height)
		return self

	def avg_color(self):
		""" 
		Get an average color of the image
		"""
		if self.color_scheme == GRAY:
			return np.mean(self.image)
		elif self.color_scheme == BGR:
			b, g, r = self._split()
			return int(np.mean(b)), int(np.mean(g)), int(np.mean(r))
		elif self.color_scheme == HSV:
			h, s, v = self.copy().hsv()._split()
			return int(np.mean(h)), int(np.mean(s)), int(np.mean(v))

	def _split(self):
		return cv2.split(self.image)

	def split_chanels(self):
		return self._split()

	def split(self):
		return list(map(lambda ch: Image.from_arr(ch, scheme=GRAY), self._split()))

	def change_contrast(self, level):
		""" 
		Change contarst of the self.image

		Parameters: 
			level (float) : New contrast (0 - 255)

		Returns: 
			self: Image

		"""
		frame = np.int16(self.image)
		frame = frame * (level / 127 + 1) - level
		frame = np.clip(frame, 0, 255)
		self.image = np.uint8(frame)
		return self

	def thresh(self, level, mode=cv2.THRESH_BINARY, max_val=255):
		""" 
		Global thresholding

		Parameters: 
			level (int)   : Threshold level
			max_val (int) : Value to fill
			mode (int)    : Threshold mode (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, etc.)

		Returns: 
			self: Image

		"""
		if self.color_scheme != GRAY:
			self.to_gray()
		self.image = cv2.threshold(self.image, level, max_val, mode)[1]
		return self

	def thresh_sauvola(self, win_size=15, k=0.2, r=None):
		"""
		Sauvola local thresholding
		"""
		self.to_gray()
		mask = skimage.filters.threshold_sauvola(self.image, window_size=win_size, k=k, r=r)
		self.image = np.uint8(255 * (self.image > mask))
		return self

	def thresh_niblack(self, win_size=15, k=0.2):
		"""
		Niblack local thresholding
		"""
		self.to_gray()
		mask = skimage.filters.threshold_niblack(self.image, window_size=win_size, k=k)
		self.image = np.uint8(255 * (self.image > mask))
		return self

	def thresh_li(self):
		"""
		Li global thresholding
		"""
		self.to_gray()
		mask = skimage.filters.threshold_li(self.image)
		self.image = np.uint8(255 * (self.image > mask))
		return self

	def thresh_isodata(self):
		self.to_gray()
		mask = skimage.filters.threshold_isodata(self.image)
		self.image = np.uint8(255 * (self.image > mask))
		return self

	def thresh_bradley(self, S_k=8, T=15.0):
		# TODO: Make faster
		"""
		Bradley local thresholding
		SUPER SLOW!!!
		"""
		self.to_gray()
		img = self.image
		h, w = self.shape[:2]

		S = w / S_k
		s2 = S / 2
		T = 15.0

		int_img = np.zeros_like(img, dtype=np.uint32)
		for col in range(w):
			for row in range(h):
				int_img[row, col] = img[0:row, 0:col].sum()

		self.image = np.zeros_like(img)

		for col in range(w):
			for row in range(h):
				y0 = int(max(row - s2, 0))
				y1 = int(min(row + s2, h - 1))
				x0 = int(max(col - s2, 0))
				x1 = int(min(col + s2, w - 1))
				count = (y1 - y0) * (x1 - x0)
				sum_ = int_img[y1, x1] - int_img[y0, x1] - int_img[y1, x0] + int_img[y0, x0]
				self.image[row, col] = 0 if img[row, col] * count < sum_ * (100. - T) / 100. else 255
		print('calc')
		return self

	def adaptive_box_thresh(self, winSize, ratio=0.):
		self.to_gray()
		img_smooth = cv2.boxFilter(self.image, cv2.CV_32FC1, winSize)
		out = self.image - (1.0 - ratio) * img_smooth
		out[out >= 0] = 255
		out[out < 0] = 0
		self.image = np.uint8(out)
		return self

	def adaptive_thresh(self, block_size, C, max_val=255, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
						thresh_type=cv2.THRESH_BINARY):
		if self.color_scheme != GRAY:
			self.to_gray()
		self.image = cv2.adaptiveThreshold(self.image, max_val, method, thresh_type, block_size, C)
		return self

	def local_adaptive_thresh(self, block_size):
		if self.color_scheme != GRAY:
			self.to_gray()
		self.image = np.uint8(255 * (self.image > skimage.filters.threshold_local(self.image, block_size, offset=10)))
		return self

	def split_blocks(self, a, b, force=False):
		arr = self.image.copy()
		matrix = []
		for x in range(0, self.width, a):
			row = []
			for y in range(0, self.height, b):
				row.append(self.copy().crop(x, y, x + a, y + b))
			matrix.append(row)
		return np.rot90(np.array(matrix))[::-1].tolist()

	def skeletonize_medial(self):
		"""
		Skeletonization using medial axis
		"""
		self.to_gray()
		skel, distance = skimage.morphology.medial_axis(self.image, return_distance=True)
		self.image = np.uint8(255 * distance * skel)
		return self

	def skeletonize_sk(self):
		"""
		Classic skeletonization
		"""
		self.to_gray()
		self.image = np.uint8(255 * skimage.morphology.skeletonize(self.image / 255.))
		return self

	def skeletonize(self, size=(3, 3)):
		self.to_gray()
		self.image = imutils.skeletonize(self.image, size=size)
		return self

	def sobel_2d(self):
		"""
		Two demensional Sobel operator
		"""
		self.to_gray()
		self.image = np.uint8(255 * skimage.filters.sobel(self.image))
		return self

	def sobel(self, x, y, size, scale=1, delta=0, border_type=cv2.BORDER_CONSTANT):
		"""
		Sobel operator
		"""
		self.to_gray()
		self.color_scheme = GRAY
		self.image = cv2.Sobel(self.image, cv2.CV_16S, x, y, ksize=size, scale=scale, delta=delta,
							   borderType=border_type)
		return self

	def canny(self, lower, upper):
		"""
		Canny edge detector
		"""
		self.to_gray()
		self.image = cv2.Canny(self.img, lower, upper)
		return self

	def laplace(self, ksize=3):
		self.to_gray()
		self.image = np.uint8(255 * skimage.filters.laplace(self.image, ksize=ksize))
		return self

	def get(self, x, y):
		"""
		Get pixel color at (x,y)
		"""
		if self.color_scheme == HSV:
			return cv2.cvtColor(np.uint8([[self.image[y, x]]]), cv2.COLOR_BGR2HSV)[0][0]
		return self.image[y, x]

	def size(self):
		"""
		Returns width, height of the image
		"""
		return self.image.shape[1], self.image.shape[0]

	def polygon(self, points, color=RED, thickness=1, is_closed=True):
		""" 
		Draw a polygon

		Parameters: 
			points (list)    : List of polygon points
			color (tuple)    : Fill/Stroke color
			thickness (int)  : Stroke thickness (will fill the poly if < 0)
			is_closed (bool) : Close the poly or not

		Returns: 
			self: Image

		"""
		if type(points) == list or type(points) == tuple:
			points = np.array([points], dtype=np.int32)
		if thickness < 0:
			self.image = cv2.fillPoly(self.image, points, color)
		else:
			self.image = cv2.polylines(self.image, points, is_closed, color)
		return self

	def line(self, a, b, color=RED, thickness=1, mode=cv2.LINE_AA):
		""" 
		Draw a line

		Parameters: 
			a (tuple)       : Start point
			b (tuple)       : End point
			thickness (int) : Stroke thickness
			color (tuple)   : Fill/Stroke color
			mode (int)      : Line type

		Returns: 
			self: Image

		"""
		self.image = cv2.line(self.image, tuple(a), tuple(b), color, thickness, mode)
		return self

	def ellipse(self, center, width, height, color=RED, thickness=1):
		""" 
		Draw an ellipse

		Parameters: 
			center (tuple)  : Ellipse center
			width (int)     : Ellipse width
			height (int)    : Ellipse height
			thickness (int) : Stroke thickness (will fill if < 0)
			color (tuple)   : Fill/Stroke color

		Returns: 
			self: Image

		"""
		self.image = cv2.ellipse(self.image, center, (width, height), 0, 0, 360, color=color, thickness=thickness)
		return self

	def circle(self, center, radius, color=RED, thickness=1):
		""" 
		Draw a circle

		Parameters: 
			center (tuple)  : Circle center
			radius (int)    : Circle radius
			thickness (int) : Stroke thickness (will fill if < 0)
			color (tuple)   : Fill/Stroke color

		Returns: 
			self: Image

		"""
		self.image = cv2.circle(self.image, center, radius, color, thickness=thickness)
		return self

	def rect(self, a, b, color=RED, thickness=1, is_size=False):
		b = ((a[0] + b[0]) if is_size else b[0], (a[1] + b[1]) if is_size else b[1])
		self.image = cv2.rectangle(self.image, a, b, color, thickness=thickness)
		return self

	def text(self, message, pos, font=cv2.FONT_HERSHEY_PLAIN, scale=1, line=1, thickness=1, color=(255,) * 3):
		self.image = cv2.putText(self.image, message, pos, font, scale, color, thickness=thickness, lineType=line)
		return self

	def find_contours(self, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE, with_hierarchy=False):
		_, cnts, hierarchy = cv2.findContours(self.image, mode, method)
		contours = []
		for cnt in cnts:
			contours.append(Contour(cnt))
		return (ContourSet(contours), hierarchy) if with_hierarchy else ContourSet(contours)

	def draw_contours(self, contours, color=RED, thickness=1, index=-1, random_color=False):
		if not (type(contours) in [ContourSet, tuple, list]):
			contours = [contours]
		if not random_color:
			self.image = cv2.drawContours(self.image, np.array(list(map(lambda c: c.contour, contours))), index,
										  color=color,
										  thickness=thickness)
		elif index == -1:
			for cnt in contours:
				color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
				self.image = cv2.drawContours(self.image, [cnt.contour], index, color=color, thickness=thickness)
		return self

	def blur(self, ksize):
		self.image = cv2.blur(self.image, ksize)
		return self

	def gaussian(self, size, sigma_x, sigma_y=0):
		self.image = cv2.GaussianBlur(self.image, size, sigma_x, sigmaY=sigma_y)
		return self

	def median(self, alpha):
		self.image = cv2.medianBlur(self.image, alpha)
		return self

	def bilateral(self, d, sigma_color, sigma_space):
		self.image = cv2.bilateralFilter(self.image, d, sigma_color, sigma_space)
		return self

	def box_filter(self, winSize):
		self.image = cv2.boxFilter(self.image, cv2.CV_32FC1, winSize)
		return self

	def filter(self, depth, kernel):
		self.image = cv2.filter2D(self.image, depth, kernel)
		return self

	def blur_pyr(self, num_levels):
		lower = self.image.copy()
		for i in range(num_levels):
			lower = cv2.pyrDown(lower)
		for i in range(num_levels):
			lower = cv2.pyrUp(lower)
		self.image = lower
		return self

	def overlay(self, other, alpha=0.5, pos=(0, 0), fill=False):
		other = other.copy()
		if fill:
			other = other.resize(width=other.size()[0], height=other.size()[1])
			pos = (0, 0)

		overlay = self.copy().put(other, pos[0], pos[1])
		self.image = cv2.addWeighted(self.image, 1, overlay.image, alpha, 0)
		return self

	def erode(self, kernel, iterations=1):
		"""
		Morphology erosion
		"""
		if type(kernel) in [tuple, list]:
			kernel = np.ones(tuple(kernel))
		self.image = cv2.erode(self.image, kernel, iterations=iterations)
		return self

	def dilate(self, kernel, iterations=1):
		"""
		Morphology dilation
		"""
		if type(kernel) in [tuple, list]:
			kernel = np.ones(tuple(kernel))
		self.image = cv2.dilate(self.image, kernel, iterations=iterations)
		return self

	def morph_gradient(self, kernel):
		if type(kernel) in [tuple, list]:
			kernel = np.ones(tuple(kernel))
		self.image = cv2.morphologyEx(self.image, cv2.MORPH_GRADIENT, kernel)
		return self

	def morph_close(self, kernel):
		if type(kernel) in [tuple, list]:
			kernel = np.ones(tuple(kernel))
		self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
		return self

	def morph_open(self, kernel):
		if type(kernel) in [tuple, list]:
			kernel = np.ones(tuple(kernel))
		self.image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
		return self

	def morph_tophat(self, kernel):
		if type(kernel) in [tuple, list]:
			kernel = np.ones(tuple(kernel))
		self.image = cv2.morphologyEx(self.image, cv2.MORPH_TOPHAT, kernel)
		return self.copy()

	def morph_blackhat(self, kernel):
		if type(kernel) in [tuple, list]:
			kernel = np.ones(tuple(kernel))
		self.image = cv2.morphologyEx(self.image, cv2.MORPH_BLACKHAT, kernel)
		return self

	def put(self, another, x, y):
		another = another.bgr()
		background, overlay = self.bgr().image, another.image
		background_width = background.shape[1]
		background_height = background.shape[0]

		if x >= background_width or y >= background_height:
			return background

		h, w = overlay.shape[0], overlay.shape[1]

		if x + w > background_width:
			w = background_width - x
			overlay = overlay[:, :w]

		if y + h > background_height:
			h = background_height - y
			overlay = overlay[:h]

		if another.channels < 4:
			overlay = np.concatenate(
				[
					overlay,
					np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
				],
				axis=2,
			)

		overlay_image = overlay[..., :3]
		mask = overlay[..., 3:] / 255.0

		background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image
		self.image = background
		return self

	def denoise(self, h=10, template_size=7, search_size=21):
		if self.color_scheme == GRAY:
			self.image = cv2.fastNlMeansDenoising(self.image, None, h, template_size, search_size)
			return self.copy()
		b, g, r = self._split()
		b = cv2.fastNlMeansDenoising(b, None, h, template_size, search_size)
		g = cv2.fastNlMeansDenoising(g, None, h, template_size, search_size)
		r = cv2.fastNlMeansDenoising(r, None, h, template_size, search_size)
		self.image = cv2.merge((b, g, r))
		return self

	def sharper(self):
		kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
		self.image = cv2.filter2D(self.image, -1, kernel)
		return self

	def adjust_contrast(self, gamma):
		invGama = 1.0 / float(gamma)
		table = np.array([((i / 255.0) ** invGama) * 255 for i in np.arange(0, 256)]).astype("uint8")
		self.image = np.uint8(cv2.LUT(self.image, table))
		return self

	def get_quarter(self, x, y):
		w, h = self.size()
		w //= 2
		h //= 2
		if x > w and y < h:
			return 1
		elif x < w and y < h:
			return 2
		elif x < w and y > h:
			return 3
		elif x > w and y > h:
			return 4
		return 0

	def deltas(self, kx=1, ky=0):
		if self.color_scheme != GRAY:
			self.to_gray()
		w, h = self.size()
		buff = np.zeros_like(self.image)
		for i in range(1, w):
			for j in range(1, h):
				pix = self.get(i, j)
				dx = abs(int(pix) - int(self.get(i - kx, j - ky)))
				buff[j, i] = dx
		self.image = np.uint8(buff)
		return self

	def brightness_peaks(self, threshold):
		self.to_gray()
		buff = np.zeros_like(self.image)
		for j, a in enumerate(self.image):
			ind = [0]
			for i in range(1, len(a) - 2):
				cur, nxt = a[i], a[i + 1]
				if cur != 0 and nxt - cur >= threshold:
					buff[j, i] = 255
		self.image = np.uint8(buff)
		return self

	def DoG(self, alpha=6.6, betta=1.5, size=(0, 0)):
		self.to_gray()
		g1 = cv2.GaussianBlur(self.image, size, alpha)
		g2 = cv2.GaussianBlur(self.image, size, betta)
		self.image = g1 - g2
		return self

	def find_corners_mask(self, blockSize, ksize, k, alpha=0.01):
		self.to_gray()
		dst = cv2.cornerHarris(self.image, blockSize, ksize, k)
		mask = np.zeros_like(dst)
		mask[dst > alpha * dst.max()] = 255
		self.image = mask
		return self

	def extrema_mask(self, h=0.5):
		self.image = np.uint8(255 * skimage.morphology.extrema.h_maxima(self.image, h))
		return self

	def fill_holes(self):
		self.to_gray()
		self.image = np.uint8(255 * ndi.binary_fill_holes(self.image))
		return self

	def inpaint(self, mask, radius, mode=cv2.INPAINT_TELEA):
		self.to_bgr()
		self.image = cv2.inpaint(self.image, mask.image, radius, mode)
		return self

	def dist_transform(self, mode=cv2.DIST_L2):
		self.image = 255 - (np.uint8(ndi.distance_transform_edt(self.image)) * 3)
		return self

	def mean_shift_filter(self, spatial_radius, color_radius):
		self.image = np.uint8(cv2.pyrMeanShiftFiltering(self.image, spatial_radius, color_radius))
		return self

	def select_roi(self, win_name, fromCenter=False, showCrosshair=True):
		return cv2.selectROI(win_name, self.image, fromCenter=fromCenter, showCrosshair=showCrosshair)

	def match_template(self, template, method=cv2.TM_CCOEFF, fixed_size=False):
		gray = self.gray().image
		template = template.gray().image
		(tH, tW) = template.shape[:2]
		found = None

		if not fixed_size:
			template = cv2.Canny(template, 50, 200)
			for scale in np.linspace(0.2, 1.0, 40)[::-1]:
				resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
				r = gray.shape[1] / float(resized.shape[1])

				if resized.shape[0] < tH or resized.shape[1] < tW:
					return (-1, -1), (-1, -1)

				edged = cv2.Canny(resized, 50, 200)
				result = cv2.matchTemplate(edged, template, method)
				(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

				if found is None:
					found = (maxVal, maxLoc, r)
				if maxVal > found[0]:
					found = (maxVal, maxLoc, r)

			(_, maxLoc, r) = found
			startX, startY = (int(maxLoc[0] * r), int(maxLoc[1] * r))
			endX, endY = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
			return (startX, startY), (endX - startX, endY - startY)
		else:
			res = cv2.matchTemplate(gray, template, method)
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
			if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
				top_left = min_loc
			else:
				top_left = max_loc
			bottom_right = (top_left[0] + tW, top_left[1] + tH)
			return top_left, (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])

	def watershed(self, ksize=np.ones((3, 3)), min_dist=20):
		self.to_gray()
		thresh = self.image
		D = ndi.distance_transform_edt(thresh)

		localMax = skimage.feature.peak_local_max(D, indices=False, min_distance=min_dist, labels=thresh)
		markers = ndi.label(localMax, structure=ksize)[0]
		labels = skimage.morphology.watershed(-D, markers, mask=thresh)

		contours = ContourSet([])

		for label in np.unique(labels):
			if label == 0:
				continue

			mask = np.zeros(thresh.shape, dtype="uint8")
			mask[labels == label] = 255

			cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
									cv2.CHAIN_APPROX_SIMPLE)
			cnts = imutils.grab_contours(cnts)
			c = max(cnts, key=cv2.contourArea)
			contours.push(Contour(c))

		return contours

	def white_balance(self):
		original_scheme = self.color_scheme
		self.to_bgr()
		result = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
		avg_a = np.average(result[:, :, 1])
		avg_b = np.average(result[:, :, 2])
		result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
		result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
		self.image = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
		if original_scheme == GRAY: self.to_gray()
		if original_scheme == HSV: self.to_hsv()
		return self


	def hist(self):
		if self.color_scheme == GRAY:
			return cv2.calcHist([self.img],[0],None,[256],[0,256])
		else:
			return np.array([cv2.calcHist([channel],[0],None,[256],[0,256]) for channel in self._split()])

	def equalize_hist(self):
		if self.color_scheme == GRAY:
			self.image = cv2.equalizeHist(self.image)
		else:
			img_yuv = cv2.cvtColor(self.image if self.color_scheme == BGR else cv2.cvtColor(self.image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2YUV )
			img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
			self.image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
			if self.color_scheme == HSV:
				self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
		return self

	def find_barcode(self, draw=False):
		from pyzbar import pyzbar
		barcodes = pyzbar.decode(self.image)
		data = []
		for barcode in barcodes:
			barcodeData = barcode.data.decode()
			barcodeType = barcode.type
			(x, y, w, h) = barcode.rect
			data.append({'text': barcodeData, 'type': str(barcodeType), 'rect': (x, y, w, h)})
			if draw:
				cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 2)
				text = "{} ({})".format(barcodeData, barcodeType)
				cv2.putText(self.image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

		return data


	def diff(self, other):
		self.image = np.uint8(cv2.absdiff(self.img, other.img))
		return self

	def k_mean(self, k):
		image = self.image
		pixel_values = image.reshape((-1, 3))
		pixel_values = np.float32(pixel_values)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
		_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
		centers = np.uint8(centers)
		labels = labels.flatten().flatten()

		segmented_image = centers[labels]
		segmented_image = segmented_image.reshape(image.shape)

		mask = labels
		mask = mask.reshape(image.shape[:2])

		self.image = segmented_image
		return self

	def hog(self, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), normalization=(0, 10), visualize=True):
		from skimage.feature import hog
		from skimage import exposure

		pack = hog(self.image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=visualize, multichannel=(self.color_scheme != GRAY))
		if visualize:
			array, image = pack
			self.image = np.uint8(255*exposure.rescale_intensity(image, in_range=normalization))
			self.color_scheme = GRAY
		else:
			array = pack
		return array, self

	@property
	def channels(self):
		if len(self.image.shape) >= 3:
			return self.image.shape[2]
		return 1

	@property
	def shape(self):
		return self.image.shape

	@property
	def img(self):
		return self.image

	@property
	def width(self):
		return self.image.shape[1]

	@property
	def height(self):
		return self.image.shape[0]

	@property
	def dtype(self):
		return self.image.dtype

	@staticmethod
	def from_arr(arr, scheme=BGR):
		tmp = Image()
		tmp.image = arr.copy()
		tmp.color_scheme = scheme
		return tmp

	@staticmethod
	def load(filename, flags=None):
		if flags is None:
			return Image.from_arr(cv2.imread(filename))
		return Image.from_arr(cv2.imread(filename))

	@staticmethod
	def merge(images, vertical=False, interpolation=cv2.INTER_CUBIC):
		result = None
		if vertical:
			w_min = min(im.shape[1] for im in images)
			im_list_resize = [
				imutils.resize(im.image, width=w_min) for
				im in
				images]
			result = cv2.vconcat(im_list_resize)
		else:
			h_min = min(im.shape[0] for im in images)
			im_list_resize = [
				cv2.resize(im.image, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
				for im in images]
			result = cv2.hconcat(im_list_resize)
		output = Image.from_arr(result)
		output.color_scheme = images[0].color_scheme
		return output

	@staticmethod
	def filled_mask(width, height):
		res = Image.from_arr(255 * np.ones((height, width), dtype=np.uint8))
		res.color_scheme = GRAY
		return res

	@staticmethod
	def filled_mask_like(target):
		res = Image.from_arr(255 * np.ones((target.img.shape[0], target.img.shape[1]), dtype=np.uint8))
		res.color_scheme = GRAY
		return res

	@staticmethod
	def blank_mask(width, height):
		res = Image.from_arr(0 * np.ones((height, width), dtype=np.uint8))
		res.color_scheme = GRAY
		return res

	@staticmethod
	def blank_mask_like(target):
		res = Image.from_arr(0 * np.ones((target.img.shape[0], target.img.shape[1]), dtype=np.uint8))
		res.color_scheme = GRAY
		return res

	def save(self, filename, flags=None):
		if flags is None:
			cv2.imwrite(filename, self.image)
		cv2.imwrite(filename, self.image, flags)

	def copy(self):
		return Image.from_arr(self.image, scheme=self.color_scheme)

	def invert(self):
		self.__invert__()
		return self

	def __invert__(self):
		self.image = ~self.image
		return self

	def __sub__(self, other):
		if other.__class__ == Image:
			res = self.image - other.image
		else:
			res = self.image - other
		return Image.from_arr(res)

	def __rsub__(self, other):
		if other.__class__ == Image:
			res = other.image - self.image
		else:
			res = other - self.image
		return Image.from_arr(res)

	def __add__(self, other):
		if other.__class__ == Image:
			res = self.image + other.image
		else:
			res = self.image + other
		return Image.from_arr(res)

	def __truediv__(self, other):
		res = self.image / other
		return Image.from_arr(res)

	def __mul__(self, other):
		tmp = self.copy()
		tmp.image = self.image * other
		return tmp

	def __rmul__(self, other):
		return self.__mul__(other)

	def __sub__(self, other):
		tmp = self.copy()
		if other.__class__ == Image:
			tmp.image = cv2.absdiff(self.image, other.image)
			return tmp
		tmp.image = self.image - other
		return tmp

	def __rsub__(self, other):
		tmp = self.copy()
		if type(other) == int:
			tmp.image = other - self.image
			return tmp
		else:
			raise SystemError(f'Unsopported method \'-\' between {str(type(other))} and Image')

	def __getitem__(self, val):
		res = self.copy()
		if type(val) == slice:
			res.image = res.image[val]
		elif len(val) == 2:
			res.image = res.image[val[0], val[1]]
		elif len(val) == 1:
			res.image = res.image[val]
		return res

	def __repr__(self):
		return f'<Image shape: {self.shape}, color_scheme: {["BGR", "GRAY", "HSV"][self.color_scheme]}>'

	def __str__(self):
		return self.__repr__()

# class ImageSet:
#
# 	def __init__(self, arr):
# 		self.array = list(arr)
#
# 	def add(self, image: Image):
# 		self.array.append(image)
#
# 	def __iter__(self):
# 		for elem in self.array:
# 			yield elem
#
# 	def __len__(self):
# 		return len(self.array)
#
# 	def __getitem__(self, ii):
# 		return self.array[ii]
#
# 	def __delitem__(self, ii):
# 		del self.array[ii]
#
# 	def __setitem__(self, ii, val):
# 		self.array[ii] = val
