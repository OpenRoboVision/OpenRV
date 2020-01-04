import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	# Developer's stuff, delete it :)

from openrv import *


def main(last, data, inst):
	frame: Image = data['frame']	# Read a frame from the webcam
	frame.resize(height=320)		# Proprtionaly resize the frame to height=320 (to process faster)

	frame.to_gray()	# Convert to GRAYSCALE

	thresh		= frame.copy().thresh(127).show('Classic') 									# Classic higher then
	thresh_inv	= frame.copy().thresh(127, mode=cv2.THRESH_BINARY_INV).show('Inversed')		# Classic lower then
	##  You can use opencv combinations of modes like (cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)

	sauvola = frame.copy().thresh_sauvola(win_size=15, k=0.2).show('Sauvola')  # Sauvola local thresholding
	niblack = frame.copy().thresh_niblack(win_size=15, k=0.2).show('Niblack')  # Niblack local thresholding
	li 		= frame.copy().thresh_li().show('Li')						 	   # Li global thresholding
	box		= frame.copy().adaptive_box_thresh((15, 15)).show('Box thresh')	   # Thresholding using diff of cv2.boxFilter
	isodata = frame.copy().thresh_isodata().show('Isodata')

	adaptive = frame.copy().adaptive_thresh(11, 2, method=cv2.ADAPTIVE_THRESH_MEAN_C)  # OpenCV standart adaptiveThreshold
	adaptive.show('Adaptive')

if __name__ == '__main__':
	app = App(src=0)	# Create a new application with a webcam input
	app.start([main])	# Start your processing loop
