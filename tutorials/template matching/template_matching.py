import sys
sys.path.append('/Users/kivicode/Documents/GitHub/OpenRV')	## Developer's stuff, delete it :)

from openrv import *



def main(last, data, inst):
	template: Image = Image.load('logo.png')
	original: Image = Image.load('warcarft.jpg')


	# print(original.match_template)
	pos, size = original.match_template(template, fixed_size=True)

	original.rect(pos, size, is_size=True, color=GREEN, thickness=2)

	original.show()
	template.show('Template')


if __name__ == '__main__':
	app = App()
	app.start([main])	# Start your processing loop