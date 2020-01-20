from env import *
import numpy as np
import matplotlib.pyplot as plt

from dqn import DeepQNetwork

# measure points
# wafer range x^2+y^2<=4
P=np.array([[0.2,0.2],[1,1],[1,-1],[-1,1],[-1,-1]])

EPISODE=1000
MAXLEN=200
ACTIONSTEP=0.01

A_R=[]

class My_Env(Env):
    def __init__(self,p):
        super().__init__(p)
        self.step_value=ACTIONSTEP
        self.maxlen=MAXLEN


if __name__ == "__main__":
    env=My_Env(P)
    RL = DeepQNetwork(env.action_space, env.observation_space,
                    learning_rate=0.01,
                    reward_decay=0.9,
                    e_greedy=0.95,
                    replace_target_iter=500,
                    memory_size=1000,
                    # output_graph=True,
                    e_greedy_increment=0.01
                    )

    step = 0
    for i in range(EPISODE):
        accu_reward=0
        env.reset()
        env.step(np.random.randint(0,env.action_space))
        observation=env.observation()
        while 1:
            
            action=RL.choose_action(observation)
            
            reward, observation_ =env.step(action)

            accu_reward+=reward
                
            RL.store_transition(observation, action, reward, observation_)

            observation=observation_

            if (step > 200) and (step % 5 == 0):
                    RL.learn()

            if env.terminate():
                print("EPISODE: {}\ncurrent reward: {}\ncurrent state: {}".format(i,env.reward(),env.S))
                A_R.append(accu_reward)
                break
            step+=1
    RL.plot_cost()
    print("end training")
    plt.plot(range(len(A_R)),A_R)
    plt.savefig("reward.png")
    TEST_EPISODE=5
    for i in range(TEST_EPISODE):
        env.reset()
        env.step(np.random.randint(0,env.action_space))
        observation=env.observation()
        while 1:
            
            action=RL.choose_action(observation)
            
            reward, observation_ =env.step(action)

            observation=observation_

            env.plot_output()
            if env.terminate():
                if env.n_step<MAXLEN:
                    print("episode{}'s para out of range".format(i))
                    print(env.S)
                break
    
        
    