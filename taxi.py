import gym
import random
import numpy as np

random.seed(1)

def choose_action(env,Q,numa,alpha,gamma):
	cur_state = env.reset()
	point= 0

	while True:
		env.render()

		#if random.uniform(0,1) < 0.05:
		#linear exp heat	action = env.action_space.sample()
		#else:
		action = np.argmax(Q[cur_state,:]+random.random()) 
		new_state, reward, done, info = env.step(action)
		point += reward

		Q[cur_state, action] = (1 - alpha) * Q[cur_state, action] +alpha * (reward + gamma * np.max(Q[new_state,:])) 
		cur_state = new_state 
		
		if done:
			break

	return point,Q

def fitness(score_list, window_size=100, threshold=8):
	for ep in range(len(score_list)-window_size):
		window = score_list[ep:ep+window_size]
		if sum(window)/window_size >= threshold:
			return ep
	return None

if __name__ == "__main__":
	env = gym.make('Taxi-v2')
	env.seed(42)
	Q = np.zeros(shape=(500,6)) 
	alpha=0.8
	gamma=0.9
	score = []
	for i_episode in range(1000):
		point,Q = choose_action(env,Q,env.action_space.n,alpha,gamma)   
		score.append(point)
	print(fitness(score))
		