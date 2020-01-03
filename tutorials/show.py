import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	## Developer's stuff, delete it :)

from openrv import *


def main(last, data, inst):
	frame: Image = data['frame']	# Read current frame from the webcam
	frame.resize(height=320)		# Proprtionaly resize the frame to height=320 (to process faster)

	frame.show()							# Default window name if 'Frame'
	frame.show('Custom win name')			# You can set your custom window name
	frame.show('Name with id. id =', 0)		# You add an index(postfix) to the window name
	frame.show('Name with id. id =', [1,2])	# Anything that can be converted to string, can be an index


if __name__ == '__main__':
	app = App(src=0)	# Create a new application with a webcam input
	app.start([main])	# Start a processing loop (you can pass >=1 functions. The will be executed one by one)