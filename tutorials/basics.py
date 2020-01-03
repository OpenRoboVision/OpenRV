import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	# Developer's stuff, delete it :)

from openrv import *	# The most convenient way to import the lib


def changed(*args):
	print('Something has changed!!!')


def main_1(last, data, inst):
	frame: Image = data['frame']	# Read current frame from the webcam
	"""
	# But you can set src=-1 and read an image from a file 
	frame: Image = Image.load('/path/to/your/image')

	# And then save it
	frame.save('/new/file/name')
	"""

	gray = frame.gray()
	return gray  # Pass it to the next function in the loop


def main_2(last, data, inst):
	gray = last	 # 'last' conteins the previous returned value

	# 'inst' contains an instance of the parent App
	a = inst.get_trackbar('Frame with a trackbar', 'A')  # You can read you trackbar value bay it's window name and it's own name
	b = inst.get_trackbar('Frame with a trackbar', 'B')
	print(f'A: {a}', f'B: {b}')

	gray.show('It Works! XD')


if __name__ == '__main__':
	app = App(src=0)	# Create a new application with a webcam input (0 - camera index, also can be a video file path)
	
	app.add_trackbar('Frame with a trackbar', 'A', 0, 255, action=changed)  # You cann add lots of trackbars to your windows
	app.add_trackbar('Frame with a trackbar', 'B', 0, 255, action=changed)
	
	app.start([main_1, main_2])	# Start a processing loop (you can pass >=1 functions. The will be executed one by one)
