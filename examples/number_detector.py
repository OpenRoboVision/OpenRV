import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')

from openrv import *
import numpy as np
import cv2


def main(last, data, inst):
	frame: Image = data['frame']
	frame.resize(height=400)

	mask = frame.copy().gray()

	mask.adaptive_threshold((121,121), 0)
	mask.gaussian((31, 31), 10).median(31).thresh(50).invert().erode(np.ones((9, 9)))

	contours = mask.find_contours(method=cv2.CHAIN_APPROX_SIMPLE, mode=cv2.RETR_EXTERNAL)
	contours.expression(lambda a: a.as_bbox()).filter_area(min_area=5000).sort(Contour.get_x)

	mask.dilate(np.ones((14, 14)))

	submasks = []
	compares = []
	dxx = []
	for i, cnt in enumerate(contours):
		dists = []
		for j, _cnt in enumerate(contours):
			if (j, i) in compares:
				continue
			dx, dy = abs(_cnt.x - cnt.x), abs(_cnt.y - cnt.y)

			if dx < 80 and dy > 100:
				compares.append((i, j))
			dists.append((dx, dy))

		x, y, w, h = cnt.bbox()
		crop = mask.copy().crop(x, y, x+w, y+h).dilate(np.ones((5,5)))
		submasks.append(crop)

	insertions = []
	for pair in compares[::-1]:
		a, b = pair
		img = Image.merge([submasks[b], submasks[a]], vertical=True)
		img.gaussian((71,71), 5).median(31).thresh(127)
		insertions += [(a, img)]
		del submasks[b]
		del submasks[a]

	for el in insertions:
		submasks.insert(el[0], el[1])

	mask.show('Mask')
	Image.merge(submasks).show('Numbers')
	frame.draw_contours(contours).show()
	


if __name__ == '__main__':
	app = App(src=0)
	app.start([main])
