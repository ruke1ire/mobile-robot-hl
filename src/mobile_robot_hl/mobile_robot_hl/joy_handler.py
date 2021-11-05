import pygame
from pygame.locals import *
from enum import Enum
from threading import Thread
import time

class JoyHandler:
	def __init__(self, max_linear_vel=None, max_angular_vel=None):
		self.state = {}
		for interface_type in InterfaceType:
			self.state[interface_type.name] = False

		self.state[InterfaceType.LINEAR_VELOCITY.name] = 0.0
		self.state[InterfaceType.ANGULAR_VELOCITY.name] = 0.0

		if(max_linear_vel == None):
			self.max_linear_vel = 1.0
		else:
			self.max_linear_vel = abs(max_linear_vel)

		if(max_angular_vel == None):
			self.max_angular_vel = 1.0
		else:
			self.max_angular_vel = abs(max_angular_vel)

		pygame.joystick.init()
		self.connect()

		if self.status == True:
			pygame.init()
			self.loop = Thread(target=lambda: self.get_state_loop())
			self.loop.start()
		else:
			return

	def connect(self):
		try:
			self.joystick = pygame.joystick.Joystick(0)
			self.joystick.init()
			print('Joystick Name: ', self.joystick.get_name())
			self.status = True
		except pygame.error:
			print('No joysticks found')
			self.status = False
	
	def get_state_loop(self):
		while True:
			for e in pygame.event.get():
				if e.type == pygame.locals.JOYAXISMOTION:
					self.state[InterfaceType.LINEAR_VELOCITY.name] = -self.joystick.get_axis(4)*self.max_linear_vel
					self.state[InterfaceType.ANGULAR_VELOCITY.name] = -self.joystick.get_axis(3)*self.max_angular_vel
				elif e.type == pygame.locals.JOYBUTTONDOWN:
					if e.button in [0,1,2,3]:
						self.state[InterfaceType.STOP.name] = True
					elif e.button == 5:
						self.state[InterfaceType.START_PAUSE_TASK.name] = True
					elif e.button == 7:
						self.state[InterfaceType.TAKE_OVER_TASK.name] = True
					elif e.button == 4:
						self.state[InterfaceType.START_PAUSE_DEMO.name] = True
					elif e.button == 6:
						self.state[InterfaceType.TERMINATION_FLAG.name] = True

				elif e.type == pygame.locals.JOYBUTTONUP:
					if e.button in [0,1,2,3]:
						self.state[InterfaceType.STOP.name] = False
					elif e.button == 5:
						self.state[InterfaceType.START_PAUSE_TASK.name] = False
					elif e.button == 7:
						self.state[InterfaceType.TAKE_OVER_TASK.name] = False
					elif e.button == 4:
						self.state[InterfaceType.START_PAUSE_DEMO.name] = False
					elif e.button == 6:
						self.state[InterfaceType.TERMINATION_FLAG.name] = False
			time.sleep(0.1)

	def get_state(self, interface_type = None):
		if(interface_type == None):
			return self.state
		else:
			return self.state[interface_type.name]

class InterfaceType(Enum):
	LINEAR_VELOCITY = 0
	ANGULAR_VELOCITY = 1
	TERMINATION_FLAG = 2
	START_PAUSE_TASK = 3
	TAKE_OVER_TASK = 4
	START_PAUSE_DEMO = 5
	STOP = 6

def main():
	joy_handler = JoyHandler()
	while True:
		print(joy_handler.get_state())

if __name__ == '__main__':
    main()