from algo import DictQ
from text_events import TextEventsEnv
import numpy as np
import matplotlib.pyplot as plt

def learn(episodes=100000,adjust=False,improve = True,pattn = 10):
    """the learing process with optional algorithm and states define """
    seed = 42
    rendering = False
    learning_rate = 0.8
    discount = 0.9
    
    env = TextEventsEnv()
    Q = DictQ(discount,learning_rate)
    env.seed(seed)
    np.random.seed(seed)

    #choose if add artificial adjustment
    if adjust:
        q_learning = getattr(Q,'q_algo_artificial_adjustment')
    else:
        q_learning = getattr(Q,'q_algo')

    #choose if use improved algorithm
    if improve: 
        create_new_k = getattr(Q,'create_new_k_begin')
    else:
        create_new_k = getattr(Q,'create_new_k_patten') 

    reward_list = []
    for i in range(episodes):
        begin = env.reset()
        if improve:
             pattn = begin
        new_k = create_new_k(begin,pattn)
        score = 0

        while True:
            if rendering:
                env.render()   
            k = new_k 

            action = Q.choose_action(k)
            observation, reward, done, info = env.step(action)
            new_k = create_new_k(observation,pattn)
            q_learning(k,action,reward,observation,info,new_k)

            score += reward	
            if done:
                break
        reward_list.append(score)
    env.close()
    plt.plot(reward_list)
    plt.show()
    print("Done")
    
if __name__ == '__main__':
	learn(100000,improve=False)