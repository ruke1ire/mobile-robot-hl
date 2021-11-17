from abc import ABC, abstractmethod
from enum import Enum

class DataType(Enum):
	int = 0
	string = 1
	image = 2
	tensor = 3

class Logger(ABC):
	'''
	This is an abstract class for logging data.
	'''

	@abstractmethod
	def log(self, data_type, data, key = None):
		'''
		This function logs the various types of data.

		Keyword arguments:
		data_type -- <DataType> The data type of the data
		data -- The data to be logged
		key -- An optional key/name of the data

		'''
		pass

class EmptyLogger(Logger):
	def log(self, data_type, data, key = None):
		return

class PrintLogger(Logger):
	def log(self, data_type, data, key = None):
		if key is not None:
			key = str(key)+': '
		else:
			key = ''

		if(data_type == DataType.int):
			print(key+str(data))
		elif(data_type == DataType.string):
			print(key+data)
		elif(data_type == DataType.image):
			print(key+str(data))
		elif(data_type == DataType.tensor):
			print(key+str(data))
