import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	# Developer's stuff, delete it :)

from openrv import *


def main(last, data, inst):
	frame: Image = data['frame']	# Read current frame from the webcam

	print(f'Number of chanels: {frame.channels}')
	print(f'Image shape: {frame.shape}')
	print(f'Image width: {frame.width}, height: {frame.height}')

	exit()


if __name__ == '__main__':
	app = App(src=0)	# Create a new application with a webcam input
	app.start([main])	# Start your processing loop
