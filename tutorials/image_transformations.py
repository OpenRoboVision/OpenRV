import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	## Developer's stuff, delete it :)

from openrv import *


def main(last, data, inst):
	frame: Image = data['frame']	# Read current frame from the webcam
	frame.resize(height=320)		# Proprtionaly resize the frame to height=320 (to process faster)

	rotated		  = frame.copy().rotate(45)				# You can rotate an image without window resizing
	rotated_bound = frame.copy().rotate(45, bound=True)	# You can rotate an image with bound window resizing

	crop = frame.copy().crop(100, 100, 250, 200)	# You can crop a pert from the image (x1, y1, x2, y2)

	rotated.show('Rotation (Basic)')
	rotated_bound.show('Rotation (Bound)')
	crop.show('Cropping')


if __name__ == '__main__':
	app = App(src=0)	# Create a new application with a webcam input
	app.start([main])	# Start your processing loop