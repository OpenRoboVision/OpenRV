import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	# Developer's stuff, delete it :)

from openrv import *


def main(last, data, inst):
	frame: Image = data['frame']	# Read current frame from the webcam
	frame.resize(height=480)		# Proprtionaly resize the frame to height=480 (to process faster)

	gray = frame.gray()

	dist                   = gray.copy().thresh_li().dist_transform()				# Distance Transform filter
	mean_shift             = frame.copy().mean_shift_filter(11, 11)				# Mean Shift filter
	difference_of_gaussian = gray.copy().DoG(alpha=6.6, betta=1.5, size=(3,3))	# Difference of Gaussian blurs
	deltas                 = gray.copy().deltas(kx=1, ky=0)						# Calculates diff between current pix[x, y] and pix[x-kx, y-ky] (SLOW!!)
	sharper                = gray.copy().sharper()								# Applys the sharping filter
	contrast               = gray.copy().adjust_contrast(3)						# Contrust adjusting

	dist.show('Distance Transform')
	mean_shift.show('Mean Shift Filter')
	difference_of_gaussian.show('DoG')
	deltas.show('Deltas')
	sharper.show('Sharper')
	contrast.show('Contrust Adjusting')


if __name__ == '__main__':
	app = App(src=0)	# Create a new application with a webcam input
	app.start([main])	# Start your processing loop
