import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	# Developer's stuff, delete it :)

from openrv import *


def main(last, data, inst):
	frame: Image = data['frame']	# Read current frame from the webcam
	frame.resize(height=480)		# Proprtionaly resize the frame to height=480 (to process faster)

	gray = frame.gray()
	mask = Image.blank_mask_like(gray).rect((0, 0), (100, 100), thickness=-1, color=WHITE)

	denoise			   = gray.copy().denoise()					 # Standart OpenCV fastNlMeansDenoising
	corners 		   = gray.copy().find_corners_mask(15, 5, 3) # Generates mask of corner points
	extrema_mask	   = gray.copy().extrema_mask()				 # Generates mask of extrema points
	brightness_peaks   = gray.copy().brightness_peaks(127) 		 # Where difference betwwen neighbors > threshold (SLOW!!)
	inpainting	  	   = gray.copy().inpaint(mask, 5)			 # Standart OpenCV inpaint
	hole_filling	   = gray.copy().thresh_li().fill_holes()	 # Auto fill holes

	color_mask		   = frame.hsv().find_color(base_color=(127, 127, 127)) # Generates a color mask
	aruco_masrkers	   = frame.gray().find_aruco()							# (corners, ids) of ArUco

	denoise.show('Denoising')
	corners.show('Corners')
	extrema_mask.show('Extrema Points')
	brightness_peaks.show('Bright Peaks')
	inpainting.show('Inpaint')
	hole_filling.show('Fill Holes')

	color_mask.show('Color Mask')
	frame.copy().draw_aruco(aruco_masrkers[0], aruco_masrkers[1]).show('ArUco')


if __name__ == '__main__':
	app = App(src=0)	# Create a new application with a webcam input
	app.start([main])	# Start your processing loop
