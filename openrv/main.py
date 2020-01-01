from app import App
from image import Image
import numpy as np
import cv2

def main(last, data, inst):
	frame: Image = data['frame']
	frame.resize(height=400)

	mask = frame.copy().mean_shift_filter(5, 40).gray().thresh_li().invert()
	mask.show()
	frame.show('Orig')


if __name__ == '__main__':
	app = App(src=0)
	app.start([main])
