import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	## Developer's stuff, delete it :)

from openrv import *
import random


def main():
	frame = Image.load('/Users/kivicode/Downloads/watershed_coins_01.jpg')
	orig = frame.copy()

	frame.mean_shift_filter(21, 51).to_gray().median(5).thresh(0, mode=BINARY|OTSU)  # Preprocess the image
	
	cnts = frame.watershed()  # Returns an array of found contours
	
	for cnt in cnts:
		pos, r = cnt.as_circle()
		orig.circle(pos, r, color=random.choice([RED, GREEN, BLUE, PURPLE]), thickness=-1)

	orig.show('Result')
	frame.show()

	cv2.waitKey()


if __name__ == '__main__':
	main()