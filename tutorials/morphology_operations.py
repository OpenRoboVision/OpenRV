import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	# Developer's stuff, delete it :)

from openrv import *

kernel = get_morph(cv2.MORPH_RECT, (5, 5))  # Kernel generating

def main(last, data, inst):
	frame: Image = data['frame']	# Read a frame from the webcam
	frame.resize(height=320)		# Proprtionaly resize the frame to height=320 (to process faster)

	frame.to_gray()	# Convert to GRAYSCALE
	thresh = frame.copy().thresh_niblack()

	thresh.show('Original')

	erode  = thresh.copy().erode(kernel).show('Erosion')	# Morphology erosion
	dilate = thresh.copy().dilate(kernel).show('Dilation')	# Morphology dilation

	gradient = thresh.copy().morph_gradient(kernel).show('Gradient')	# Morphology gradient
	m_close  = thresh.copy().morph_close(kernel).show('Close')			# Morphology closing
	m_open   = thresh.copy().morph_open(kernel).show('Open')			# Morphology opening
	tophat   = thresh.copy().morph_tophat(kernel).show('TopHat')		# Morphology tophat
	blackhat = thresh.copy().morph_blackhat(kernel).show('BlackHat')	# Morphology blackhat


if __name__ == '__main__':
	app = App(src=0)	# Create a new application with a webcam input
	app.start([main])	# Start your processing loop
