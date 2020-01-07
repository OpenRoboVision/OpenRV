import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	## Developer's stuff, delete it :)

from openrv import *
from keras.models import model_from_json


def load_model():
	global loaded_model
	json_file = open('model/model.json', 'r')     
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model/model.h5")


def predict_digit(image):
	image.to_gray().resize(width=28, height=28)
	data = image.img.reshape(1,1,28,28) 
	return loaded_model.predict_classes(data, verbose=0)[0]    


def main():
	frame: Image = Image.load('sudoku.jpg')
	frame.resize(height=450)

	orig = frame.copy().show('Original image')

	frame.to_gray().gaussian((5,5), 0).adaptive_thresh(5, 2).invert()
	contours = frame.find_contours(mode=cv2.RETR_LIST).sort_area(reverse=True)
	cnt = contours.filter_corners_number(4, 0.02)[0].approx(0.02)

	orig.draw_contours(cnt)

	field = orig.gray().wrap_perspective_rect(cnt.contour, 450, 450)
	field = field.adaptive_thresh(11, 2).invert().erode((3,3))

	block_size = 50
	blocks = field.split_blocks(block_size, block_size, True)

	matrix = []

	print('   Found matrix')
	print('___________________')

	for i in range(len(blocks)):
		row = []
		for j in range(len(blocks[i])):
			img = blocks[i][j]
			img.crop(5, 5, -7, -7)

			mean = int(img.copy().crop(7, 7, -7, -7).avg_color())

			if mean <= 10:
				row.append(' ')
			else:
				digit = predict_digit(img)
				row.append(str(digit))

		print('|' + '|'.join(row) + '|')


	frame.show('Filtration')
	orig.show('Field found')
	field.show('After the perspective correction')

	cv2.waitKey()



if __name__ == '__main__':
	load_model()
	main()