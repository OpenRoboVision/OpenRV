import numpy as np


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


def constrain(a, low, high):
	if a < low:
		return low
	if a > high:
		return high
	return a
