import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	## Developer's stuff, delete it :)

from openrv import *


def main(last, data, inst):
	frame: Image = data['frame']	# Read a frame from the webcam
	frame.resize(height=320)		# Proprtionaly resize the frame to height=320 (to process faster)

	gray = frame.gray()	# Get gray version of the original frame
	hsv  = frame.hsv()	# Get HSV version of the original frame

	upper_row = Image.merge([	# Concate images horizontaly
		frame,
		gray.bgr()	# All the images MUST be with the same number of chanels^ so we should convert it into BGR
	])

	lower_row = Image.merge([
		frame,
		hsv
	])

	Image.merge([ 	# Now concate two rows verticaly
		upper_row,
		lower_row
	], vertical=True).show()	# And show the result


if __name__ == '__main__':
	app = App(src=0)	# Create a new application with a webcam input
	app.start([main])	# Start a processing loop