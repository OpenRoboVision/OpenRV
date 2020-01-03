import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	# Developer's stuff, delete it :)

from openrv import *


def main(last, data, inst):
	frame: Image = data['frame']	# Read current frame from the webcam
	frame.resize(height=480)		# Proprtionaly resize the frame to height=480 (to process faster)

	mask = Image.blank_mask_like(frame)

	# Draw something white on the mask
	mask.rect((100, 100), (300, 170), is_size=True, thickness=-1, color=WHITE)
	mask.circle((300, 300), 100, thickness=-1, color=WHITE)

	masked = frame.copy().apply_mask(mask)	# Apply the mask

	frame.show()
	mask.show('Mask')
	masked.show('Masked')


if __name__ == '__main__':
	app = App(src=0)	# Create a new application with a webcam input
	app.start([main])	# Start your processing loop
