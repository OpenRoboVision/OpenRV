import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	## Developer's stuff, delete it :)

from openrv import *


def main(last, data, inst):
	frame: Image = data['frame']	# Read current frame from the webcam
	frame.resize(height=480)		# Proprtionaly resize the frame to height=480 (to process faster)

	frame.line((10, 10), (140, 100), color=RED, thickness=3)		# Draw a line from (10, 10) to (140, 100)  

	frame.rect((150, 10),  (250, 100), color=BLUE,   thickness=2)	# Draw a rect "border" from (150, 10) to (250, 100)
	frame.rect((150, 120), (250, 210), color=GREEN,  thickness=-1)	# Fill a rect from (150, 120) to (250, 210)
	frame.rect((150, 230), (100, 100), color=PURPLE, is_size=True)	# Fill a rect with top-left corner at (150, 230) and size = 100x100

	frame.circle((320, 50), 50)					# Draw a circle with center at (320, 50) and radius = 50
	frame.circle((420, 50), 50, thickness=-1)	# Fill the circle

	frame.ellipse((400, 200), 100, 50)					# Draw an ellipse with center at (400, 200) and radius_x=100, radius_y=50
	frame.ellipse((400, 300), 100, 50, thickness=-1)	# Fill the ellipse

	frame.text('Hello, OpenRV', (30, 400))	# Draw text with the top-right corner at (30, 400)
	frame.text('Hello, OpenRV', (230, 400), scale=2, thickness=2, color=BLACK)  # Draw scaled text

	frame.polygon([
		[500, 480],
		[400, 480],
		[300, 380],
		], color=BLUE, is_closed=True, thickness=-1) # Fill the polygon by points

	frame.show()


if __name__ == '__main__':
	app = App(src=0)	# Create a new application with a webcam input
	app.start([main])	# Start your processing loop