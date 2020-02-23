import cv2
import numpy as np
import openrv.utils
import math


class Contour:

	def __init__(self, contour):
		self.contour = contour

	def as_line(self):
		vx, vy, x, y = cv2.fitLine(self.contour, cv2.DIST_L2, 0, 0.01, 0.01)
		return np.array([(x, y), np.math.degrees(np.math.atan2(vx, vy))])

	def as_ellipse(self):
		return cv2.fitEllipse(self.contour)

	def as_circle(self):
		(x, y), radius = cv2.minEnclosingCircle(self.contour)
		center = (int(x), int(y))
		radius = int(radius)
		return np.array([center, radius])

	def as_rect(self):
		rct = cv2.minAreaRect(self.contour)
		box = cv2.boxPoints(rct)
		box = np.int0(box)
		self.contour = box
		return self

	def as_bbox(self):
		pts = cv2.boundingRect(self.contour)
		self.contour = np.array([
			[pts[0], pts[1]],
			[pts[0] + pts[2], pts[1]],
			[pts[0] + pts[2], pts[1] + pts[3]],
			[pts[0], pts[1] + pts[3]]
		])
		return self

	def bbox(self):
		return cv2.boundingRect(self.contour)

	def get_angle(self):
		(x, y), (w, h), angle = cv2.minAreaRect(self.contour)
		# w = utils.dist(bbox[0], bbox[1])
		# h = utils.dist(bbox[1], bbox[2])
		if w < h:
			angle += 90
		return angle

	def center(self):
		M = cv2.moments(self.contour)
		if M['m00'] == 0:
			return None
		return np.array([int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])])

	def area(self):
		return cv2.contourArea(self.contour)

	def perimeter(self):
		return cv2.arcLength(self.contour, True)

	def scale(self, factor):
		cx, cy = self.center()
		cnt_norm = self.contour - [cx, cy]
		cnt_scaled = cnt_norm * factor
		cnt_scaled = cnt_scaled + [cx, cy]
		cnt_scaled = cnt_scaled.astype(np.int32)
		self.contour = cnt_scaled
		return self

	def rotate(self, angle):
		cx, cy = self.center()
		cnt_norm = self.contour - [cx, cy]

		coordinates = cnt_norm[:, 0, :]
		xs, ys = coordinates[:, 0], coordinates[:, 1]
		thetas, rhos = utils.cart_to_pol(xs, ys)

		thetas = np.rad2deg(thetas)
		thetas = (thetas + angle) % 360
		thetas = np.deg2rad(thetas)

		xs, ys = utils.pol_to_cart(thetas, rhos)

		cnt_norm[:, 0, 0] = xs
		cnt_norm[:, 0, 1] = ys

		cnt_rotated = cnt_norm + [cx, cy]
		cnt_rotated = cnt_rotated.astype(np.int32)

		self.contour = cnt_rotated

		return self

	def get_angles(self):
		outp = []
		pts = self.contour
		for i in range(len(pts)):
			a, b, c = pts[i][0], pts[(i + 1) % len(pts)][0], pts[(i + 2) % len(pts)][0]
			outp.append(openrv.utils.angle_3pts(a, b, c))
		return outp

	def cosine_sum(self):
		angles = self.get_angles()
		return sum(list(map(lambda a: math.cos(math.radians(a)), angles)))

	def approx(self, tolerance):
		epsilon = tolerance * cv2.arcLength(self.contour, True)
		self.contour = cv2.approxPolyDP(self.contour, epsilon, True)
		return self

	def copy(self):
		return Contour(self.contour)

	@property
	def x(self):
		return self.center()[0]

	@property
	def y(self):
		return self.center()[1]

	@property
	def corners(self):
		return len(self.contour)

	def get_x(self):
		return self.center()[0]

	def get_y(self):
		return self.center()[1]

	def __repr__(self):
		return f'<Contour area: {self.area()}, ' \
			   f'perimeter: {self.perimeter()}, center: {self.center()}, angle: {self.get_angle()}>'

	def __str__(self):
		return self.__repr__()

	def __iter__(self):
		for elem in self.contour:
			yield elem

	def __len__(self):
		return len(self.contour)

	def __getitem__(self, ii):
		return self.contour[ii]

	def __delitem__(self, ii):
		del self.contour[ii]

	def __setitem__(self, ii, val):
		self.contour[ii] = val


class ContourSet:

	def __init__(self, data):
		self.array = list(data)

	def filter_area(self, min_area=0, max_area=np.math.inf):
		self.array = list(filter(lambda cnt: min_area <= cnt.area() <= max_area, self.array))
		return self

	def filter(self, expr):
		self.array = list(filter(expr, self.array))
		return self

	def filter_corners_number(self, number, approx_tolerance):
		self.array = list(filter(lambda cnt: cnt.copy().approx(approx_tolerance).corners == number, self.array))
		return self

	def map(self, expr):
		self.array = list(map(expr, self.array))
		return self

	def sort(self, key, reverse=False):
		self.array = sorted(self.array, key=key, reverse=reverse)
		return self

	def sort_area(self, reverse=False):
		self.array = sorted(self.array, key=Contour.area, reverse=reverse)
		return self

	def push(self, obj):
		self.array.append(obj)
		return self

	def pop(self, obj, index=...):
		return self.array.pop(index)

	def remove(self, index):
		del self.array[index]
		return self

	def translate(self, x, y):
		for i in range(len(self.array)):
			self.array[i].contour += [x, y]
		return self

	def copy(self):
		return ContourSet(self.array.copy())

	def __iter__(self):
		for elem in self.array:
			yield elem

	def __len__(self):
		return len(self.array)

	def __getitem__(self, ii):
		return self.array[ii]

	def __delitem__(self, ii):
		del self.array[ii]

	def __setitem__(self, ii, val):
		self.array[ii] = val

	def __repr__(self):
		return str(self.array)

	def __str__(self):
		return str(self.array)
