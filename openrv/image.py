import numpy as np
import cv2
import imutils
import cv2.aruco as aruco
from openrv.colors import *
from openrv.contour import Contour, Contours
import skimage.filters
import skimage.morphology
from scipy import ndimage as ndi


aruco_params = aruco.DetectorParameters_create()


def get_morph(elem, size):
	return cv2.getStructuringElement(elem, size)


class Image:
	cv = cv2

	def __init__(self):
		self.image = np.array([])
		self.color_scheme = BGR


	def show(self, name='Frame'):
		""" 
		Show self.image

		Will draw this image on the window. Will create it if doesn't exists

		Parameters: 
			name : Name of the window to draw on

		Returns: 
			Image: self 

		"""
		cv2.imshow(name, self.image)
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
		self.image = imutils.resize(self.image, width=width, height=height, inter=inter)
		return self


	def gray(self):
		""" 
		Creates a grayscale copy of the image

		Returns: 
			Image: GRAYSCALE copy

		"""
		if self.color_scheme == GRAY:
			return self
		res = Image.from_arr(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))
		res.color_scheme = GRAY
		return res


	def to_gray(self):
		""" 
		Convert this image to GRAYSCALE

		Returns: 
			Image: self

		"""
		inp = self.gray()
		self.image = inp.img
		self.color_scheme = inp.color_scheme


	def bgr(self):
		""" 
		Creates a BGR copy of the image

		Returns: 
			Image: BGR copy

		"""
		if self.color_scheme == BGR:
			return self
		key = cv2.COLOR_GRAY2BGR
		if self.color_scheme == HSV:
			key = cv2.COLOR_HSV2BGR
		res = Image.from_arr(cv2.cvtColor(self.image, key))
		res.color_scheme = BGR
		return res


	def to_bgr(self):
		""" 
		Convert this image to BGR

		Returns: 
			Image: self

		"""
		inp = self.bgr()
		self.image = inp.img
		self.color_scheme = inp.color_scheme


	def hsv(self):
		""" 
		Creates a HSV copy of the image

		Returns: 
			Image: grayscaled copy

		"""
		if self.color_scheme == HSV:
			return self
		res = Image.from_arr(cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV))
		res.color_scheme = HSV
		return res


	def to_hsv(self):
		""" 
		Convert this image to HSV

		Returns: 
			Image: self

		"""
		inp = self.hsv()
		self.image = inp.img
		self.color_scheme = inp.color_scheme


	def rotate(self, angle, bound=False):
		""" 
		Rotate self.image on given angle

		Parameters: 
			angle (int)  : rotatation angle (clockwise)
			bound (bool) : crop the image or not

		Returns: 
			self: Image

		"""
		if bound:
			self.image = imutils.rotate_bound(self.image, angle)
		else:
			self.image = imutils.rotate(self.image, -angle)
		return self


	def apply_mask(self, mask):
		""" 
		Apply the bitwise mask to self.image

		Parameters: 
			mask (Image) : bitwise mask to apply

		Returns: 
			self: Image

		"""
		mask = mask.copy()
		if self.channels > 2:
			mask.to_gray()
		self.image = cv2.bitwise_and(self.image, self.image, mask=mask)
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
			res.to_bgr()

		if base_color is not None:
			lower = [base_color[0] - sensitivity, base_color[1] - (sens_mult * sensitivity),
					 base_color[2] - (sens_mult * sensitivity)]
			upper = [base_color[0] + sensitivity, base_color[1] + (sens_mult * sensitivity),
					 base_color[2] + (sens_mult * sensitivity)]

		self.image = cv2.inRange(res.img, np.array(lower), np.array(upper))
		return self


	def erode(self, kernel, iterations=1):
		"""
		Morphology erosion
		"""
		self.image = cv2.erode(self.image, kernel, iterations=iterations)
		return self


	def dilate(self, kernel, iterations=1):
		"""
		Morphology dilation
		"""
		self.image = cv2.dilate(self.image, kernel, iterations=iterations)
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
		M = cv2.getPerspectiveTransform(src, dst)
		self.image = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
		return self


	def crop(self, x1, y1, x2, y2):
		""" 
		Crop a par of image
		"""
		self.image = self.image[y1:y2, x1:x2]
		return self


	def avg_color(self):
		""" 
		Get an average color of the image
		"""
		if self.color_scheme == GRAY:
			return np.mean(self.image)
		elif self.color_scheme == BGR:
			r, g, b = self._split()
			return np.mean(r), np.mean(g), np.mean(b)
		elif self.color_scheme == HSV:
			h, s, v = self.copy().hsv()._split()
			return np.mean(h), np.mean(s), np.mean(v)


	def _split(self):
		return cv2.split(self.image)


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
		self.image = 255 * np.uint8(self.image > mask)
		return self

	def thresh_niblack(self, win_size=15, k=0.2):
		"""
		Niblack local thresholding
		"""
		self.to_gray()
		mask = skimage.filters.threshold_niblack(self.image, window_size=win_size, k=k)
		self.image = 255 * np.uint8(self.image > mask)
		return self

	def thresh_li(self):
		"""
		Li global thresholding
		"""
		self.to_gray()
		mask = skimage.filters.threshold_li(self.image)
		self.image = 255 * np.uint8(self.image > mask)
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

		S = w/S_k
		s2 = S/2
		T = 15.0

		int_img = np.zeros_like(img, dtype=np.uint32)
		for col in range(w):
			for row in range(h):
				int_img[row,col] = img[0:row,0:col].sum()

		self.image = np.zeros_like(img) 
		   
		for col in range(w):
			for row in range(h):
				y0 = int(max(row-s2, 0))
				y1 = int(min(row+s2, h-1))
				x0 = int(max(col-s2, 0))
				x1 = int(min(col+s2, w-1))
				count = (y1-y0)*(x1-x0)
				sum_ = int_img[y1, x1]-int_img[y0, x1]-int_img[y1, x0]+int_img[y0, x0]
				self.image[row,col] = 0 if img[row, col]*count < sum_*(100.-T)/100. else 255
		print('calc')
		return self


	def skeletonize_medial(self):
		"""
		Skeletonization using medial axis
		"""
		self.to_gray()
		skel, distance = skimage.morphology.medial_axis(self.image, return_distance=True)
		self.image = np.uint8(distance * skel)
		return self


	def skeletonize(self):
		"""
		Classic skeletonization
		"""
		self.to_gray()
		self.image = 255 * np.uint8(skimage.morphology.skeletonize(self.image/255.))
		return self


	def sobel_2d(self, size=3, ultra_bright=False):
		"""
		Two demensional Sobel operator
		"""
		self.to_gray()
		gray = cv2.GaussianBlur(self.image, (size,) * 2, 0)
		grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=size, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
		grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=size, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
		abs_grad_x = cv2.convertScaleAbs(grad_x)
		abs_grad_y = cv2.convertScaleAbs(grad_y)
		alpha = 1.5 if ultra_bright else 0.5
		self.image = cv2.addWeighted(abs_grad_x, alpha, abs_grad_y, alpha, 0)
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
		return (contours, hierarchy) if with_hierarchy else Contours(contours)


	def draw_contours(self, contours, color=RED, thickness=1, index=-1, random_color=False):
		if not (type(contours) in [Contours, tuple, list]):
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


	def gaussian(self, size, sigma_x, sigma_y=0):
		self.image = cv2.GaussianBlur(self.image, size, sigma_x, sigmaY=sigma_y)
		return self


	def median(self, alpha):
		self.image = cv2.medianBlur(self.image, alpha)
		return self


	def bilateral(self, d, sigma_color, sigma_space):
		self.image = cv2.bilateralFilter(self.image, d, sigma_color, sigma_space)
		return self


	def filter(self, depth, kernel):
		self.image = cv2.filter2D(self.image, depth, kernel)
		return self


	def overlay(self, other, alpha=0.5):
		other = other.copy().resize(width=other.size()[0], height=other.size()[1])
		self.image = cv2.addWeighted(self.image, 1, other.image, alpha, 0)
		return self


	def morph_gradient(self, kernel):
		self.image = cv2.morphologyEx(self.image, cv2.MORPH_GRADIENT, kernel)
		return self

	def morph_close(self, kernel):
		self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
		return self

	def morph_open(self, kernel):
		self.image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
		return self

	def morph_tophat(self, kernel):
		self.image = cv2.morphologyEx(self.image, cv2.MORPH_TOPHAT, kernel)
		return self.copy()

	def morph_blackhat(self, kernel):
		self.image = cv2.morphologyEx(self.image, cv2.MORPH_BLACKHAT, kernel)
		return self

	def put(self, another, x, y):
		background, overlay = self.image, another.image
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

		if overlay.shape[2] < 4:
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
		kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
		self.image = cv2.filter2D(self.image, -1, kernel)
		return self

	def adjust_contrast(self, gamma):		
		invGama = 1.0 / float(gamma)
		table = np.array([((i / 255.0) ** invGama) * 255 for i in np.arange(0, 256)]).astype("uint8")
		self.image = np.uint8(cv2.LUT(self.image, table))
		return self


	def adaptive_thresh(self, block_size, C, max_val=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type=cv2.THRESH_BINARY):
		if self.color_scheme != GRAY:
			self.to_gray()
		self.image = cv2.adaptiveThreshold(self.image, max_val, adaptive_method, thresh_type, block_size, C)
		return self


	def get_quarter(self, x, y):
		w, h = self.size()
		if x > w and y > h:
			return 1
		elif x < w and y > h:
			return 2
		elif x < w and y < h:
			return 3
		elif x > w and y < h:
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
				dx = abs(int(pix) - int(self.get(i-1, j)))
				buff[j,i] = dx
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


	def DoG(self, alpha=6.6, betta=1.5):
		self.to_gray()
		g1 = cv2.GaussianBlur(self.image, (0,0), alpha)
		g2 = cv2.GaussianBlur(self.image, (0,0), betta)
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
		self.image = 255 * np.uint8(skimage.morphology.extrema.h_maxima(self.image, h))
		return self

	def fill_holes(self):
		self.to_gray()
		self.image = 255 * np.uint8(ndi.binary_fill_holes(self.image))
		return self

	def inpaint(self, mask, radius, mode=cv2.INPAINT_TELEA):
		self.to_gray()
		self.image = cv2.inpaint(self.image, mask, radius, mode)
		return self

	def dist_transform(self, size, mode=cv2.DIST_L2):
		self.image = np.uint8(cv2.distanceTransform(self.image, mode, size))
		return self

	def mean_shift_filter(self, spatial_radius, color_radius):
		self.image = np.uint8(cv2.pyrMeanShiftFiltering(self.image, spatial_radius, color_radius))
		return self

	@property
	def channels(self):
		return self.image.shape[2] or 1

	@property
	def shape(self):
		return self.image.shape

	@property
	def img(self):
		return self.image

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
		return Image.from_arr(result)

	@staticmethod
	def filled_mask(width, height):
		return Image.from_arr(255 * np.ones((height, width), dtype=np.uint8))

	@staticmethod
	def filled_mask_like(target):
		return Image.from_arr(255 * np.ones((target.img.shape[0], target.img.shape[1]), dtype=np.uint8))

	@staticmethod
	def blank_mask(width, height):
		return Image.from_arr(0 * np.ones((height, width), dtype=np.uint8))

	@staticmethod
	def blank_mask_like(target):
		tmp = target.copy()
		tmp.image = 0 * np.ones_like(target.img, dtype=np.uint8)
		return tmp

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

	def __repr__(self):
		return f'<Image shape: {self.shape}, color_scheme: {["BGR", "GRAY", "HSV"][self.color_scheme]}>'

	def __str__(self):
		return self.__repr__()
