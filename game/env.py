import numpy as np

GameDescription = {}
GameDescription['aliens'] = {'ascii':[".","0","1","2","A"], 
							  'mapping':[13, 3, 11, 12, 1], 
							  'state_shape':(14, 12, 32), 
							  'model_shape':[(3, 4),(6, 8),(12, 16),(12, 32)]}

class Env:
	def __init__(self, name, length):
		self.name = name
		self.length = length
		try:
			self.ascii = GameDescription[name]['ascii']
			self.mapping = GameDescription[name]['mapping']
			self.state_shape = GameDescription[name]['state_shape']
			self.model_shape = GameDescription[name]['model_shape']
		except:
			raise Exception(name + " data not implemented in env.py")

		self.map_level = np.vectorize(lambda x: self.ascii[x])

	def create_levels(self, tensor):
		lvl_array = tensor.argmax(dim=1).cpu().numpy()
		lvls = self.map_level(lvl_array).tolist()
		lvl_strs = ['\n'.join([''.join(row) for row in lvl]) for lvl in lvls]
		return lvl_strs


