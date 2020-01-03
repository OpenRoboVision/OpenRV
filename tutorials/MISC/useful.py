import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	# Developer's stuff, delete it :)

from openrv import *


def main(last, data, inst):
	frame: Image = data['frame']	# Read current frame from the webcam
	frame.resize(height=480)		# Proprtionaly resize the frame to height=480 (to process faster)

	gray = frame.gray()
	mask = Image.blank_mask(100, 100).rect((0, 0), (50, 50), thickness=-1, color=WHITE)

	frame.copy().put(mask, 10, 10).show('.put(x,y)') # You can put a smaller image on the bigger one with top-left corner at (x, y)
	frame.copy().overlay(mask, alpha=0.4, pos=(10, 10)).show('overlay') # Or make an overlayable image transparent

	frame.copy().invert().show()  # Invert image

	print(f'Quarter index:\t{frame.get_quarter(10, 10)}')  # Quarter index of the point(x, y)
	print(f'Avg color:\t{frame.avg_color()}')	# Get an average colot of the image
	print()


if __name__ == '__main__':
	app = App(src=0)	# Create a new application with a webcam input
	app.start([main])	# Start your processing loop
