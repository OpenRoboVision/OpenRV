import numpy as np
from scipy.spatial import distance as sci_dist
from collections import OrderedDict
import math

_colors_dict = OrderedDict({
		"red": (255, 0, 0),
		"green": (0, 255, 0),
		"blue": (0, 0, 255)})


def dist(a, b):
	return np.math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def remap(value, left_min, left_max, right_min, right_max):
	left_span = left_max - left_min
	right_span = right_max - right_min
	value_scaled = float(value - left_min) / float(left_span)
	return right_min + (value_scaled * right_span)


def cart_to_pol(x, y):
	theta = np.arctan2(y, x)
	rho = np.hypot(x, y)
	return theta, rho


def pol_to_cart(theta, rho):
	x = rho * np.cos(theta)
	y = rho * np.sin(theta)
	return x, y


def max_lambda(expression, objects):
	return max(objects, key=expression)


def min_lambda(expression, objects):
	return min(objects, key=expression)


def order_points(pts):
	rect = np.zeros((4, 2), dtype="float32")
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect


def angle_3pts(a, b, c):
	"""Counterclockwise angle in degrees by turning from a to c around b 
		Returns a float between 0.0 and 360.0"""
	ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
	return (ang + 360 if ang < 0 else ang) % 180


def constrain(a, low, high):
	if a < low:
		return low
	if a > high:
		return high
	return a


def draw_histogram(hist, block_process=True):
	from matplotlib import pyplot as plt
	plt.clf()
	color = ('b','g','r')
	if len(hist.shape) == 2:
		color = ('b')

	for i, col in enumerate(color):
		plt.plot(hist[i], color=col)
		plt.xlim([0,256])
	if block_process:
		plt.show()
	else:
		plt.draw()
		plt.pause(0.0000001)



def get_morph(elem, size):
	return cv2.getStructuringElement(elem, size)