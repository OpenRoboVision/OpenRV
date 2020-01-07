import serial

class Arduino(serial.Serial):

	def print(self):
		print('Hi')