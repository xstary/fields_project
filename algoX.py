import numpy as np
class DictQ(object):
	"""q-learning using q-value dictionary"""
	def __init__(self, discount_rate=0.9,learning_rate=0.8):
		super(DictQ, self).__init__()
		self.dic = {}
		self.disc = discount_rate
		self.lear = learning_rate
		self.action = ['A','B','C','D']

	def choose_action(self,k):		
		return np.argmax(self.dic[k])

	def create_new_k_patten(self,observation,n):
		"""create a new state with previous n letters and current index"""
		length = len(observation)
		if length <= n:
			new_k = (length,observation)
		else:
			new_k = (length,observation[-n:])
		
		if not self.dic.has_key(new_k):
			self.dic[new_k] = np.array([0,0,0,0])
		return new_k

	def create_new_k_begin(self,observation,begin):
		"""create a new state defined by begin string and current index"""
		length = len(observation)
		new_k = (length,begin)
		if not self.dic.has_key(new_k):
			self.dic[new_k] = np.array([0,0,0,0])
		return new_k

	def q_algo(self,k,action,reward,observation,info,new_k):
		"""original q learning algorithm"""
		self.dic[k][action] = (1 - self.lear) * self.dic[k][action] +self.lear * (reward + self.disc * np.max(self.dic[new_k]))  

	def q_algo_artificial_adjustment(self,k,action,reward,observation,info,new_k):
		"""q learning algorithm with artificial adjustment on correct action"""
		self.dic[k][action] = (1 - self.lear) * self.dic[k][action] +self.lear * (reward + self.disc * np.max(self.dic[k]))  
		correct_action = info[true_event]
		self.dic[K][self.action.index[correct_action]] += 0.5

