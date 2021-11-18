class ROSWidget():
	def __init__(self, parent):
		try:
			self.ros_node = parent.ros_node
		except:
			self.ros_node = None