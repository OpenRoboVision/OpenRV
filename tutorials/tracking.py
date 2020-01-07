import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	## Developer's stuff, delete it :)

from openrv import *


def main(last, data, inst):
	frame: Image = data['frame']	# Read current frame from the webcam
	frame.resize(height=480)		# Proprtionaly resize the frame to height=480 (to process faster)

	if data['key'] == 's':
		roi = frame.select_roi('Frame')
		inst.set_tracker('Face', frame, roi)

	ret, (pos, size) = inst.get_tracker('Face', frame)
	if ret:
		frame.rect(pos, size, is_size=True, color=GREEN, thickness=2)

	frame.show()


if __name__ == '__main__':
	app = App(src=0)	# Create a new application with a webcam input
	app.create_tracker('Face', tracker_type='csrt')
	app.start([main])	# Start your processing loop