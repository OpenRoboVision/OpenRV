import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	# Developer's stuff, delete it :)

from openrv import *


kernel = np.ones((9,9))


def main(last, data, inst):
	frame = Image.blank_mask(200, 200)

	# Draw something a the mask
	frame.rect((10, 10), (100, 100), thickness=-1, is_size=True, color=WHITE)

	classic		= frame.copy().blur((5, 5))			# Classic cv2.blur
	gaussian	= frame.copy().gaussian((5, 5), 0)	# Gaussian blur
	median 		= frame.copy().median(5)			# Median blur
	bilateral 	= frame.copy().bilateral(5, 70, 70)	# Bilaterial blur
	box 		= frame.copy().box_filter((10, 10))	# Box filter
	filtered 	= frame.copy().filter(1, kernel)	# Standart filter with custom kernel

	classic.show('Classic blur')
	gaussian.show('Gaussian blur')
	median.show('Median blur')
	bilateral.show('Bilateral blur')
	box.show('Box blur')
	filtered.show('Filtered')


if __name__ == '__main__':
	app = App()	# Create a new application
	app.start([main])	# Start your processing loop
