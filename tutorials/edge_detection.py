import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	## Developer's stuff, delete it :)

from openrv import *


def main(last, data, inst):
	frame: Image = data['frame']	# Read current frame from the webcam
	frame.resize(height=480)		# Proprtionaly resize the frame to height=480 (to process faster)

	gray = frame.gray()	# Convert to GRAYSCALE

	sobel    = gray.copy().sobel(1, 0, 5)	# Standart OpenCV Sobel operator
	sobel_2d = gray.copy().sobel_2d()		# Combination of X and Y Sobels
	canny    = gray.copy().canny(50, 200)	# Standart Canny edge detector
	laplace  = gray.copy().laplace()		# Laplacian edge detector

	sobel.show('Sobel')	
	sobel_2d.bgr().show('Sobel 2D')
	canny.show('Canny')
	laplace.show('Laplacian')


if __name__ == '__main__':
	app = App(src=0)	# Create a new application with a webcam input
	app.start([main])	# Start your processing loop