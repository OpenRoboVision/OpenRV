import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	## Developer's stuff, delete it :)

from openrv import *


def main(last, data, inst):
	frame: Image = Image.load('/Users/kivicode/Desktop/Снимок экрана 2020-01-04 в 15.19.06.png')
	# frame: Image = data['frame']	# Read current frame from the webcam
	frame.resize(height=480)		# Proprtionaly resize the frame to height=480 (to process faster)

	gray = frame.gray()			# Convert to GRAYSCALE
	thresh = gray.copy().thresh(127)	# Get a binary image to work with

	skeleton         = thresh.copy().skeletonize()			# Classic skeletonization
	skeleton_skimage = thresh.copy().skeletonize_sk()		# Classic skeletonization
	medail_skeleton  = thresh.copy().skeletonize_medial()	# Skeletonization using medial axis

	thresh.show('Orig')
	skeleton.show('Classic')
	skeleton_skimage.show('SkImage')
	medail_skeleton.show('Medial')
	


if __name__ == '__main__':
	app = App(src=0)	# Create a new application with a webcam input
	app.start([main])	# Start your processing loop