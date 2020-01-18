from env import *
import numpy as np

from dqn import DeepQNetwork

# measure points
# wafer range x^2+y^2<=4
P=np.array([[0.1,0.1],[1,1],[1,-1],[-1,1],[-1,-1]])

EPISODE=3000
MAXLEN=200
STEP=0.01
class My_Env(Env):
    def __init__(self,p):
        super().__init__(p)
        self.step_value=STEP
        self.maxlen=MAXLEN


if __name__ == "__main__":
    env=My_Env(P)
    RL = DeepQNetwork(env.action_space, env.observation_space,
                    learning_rate=0.01,
                    reward_decay=0.9,
                    e_greedy=0.9,
                    replace_target_iter=200,
                    memory_size=2000,
                    # output_graph=True,
                    e_greedy_increment=0.01
                    )

    step = 0
    for i in range(EPISODE):
        env.reset()
        env.step(np.random.randint(0,env.action_space))
        observation=env.observation()
        while 1:
            
            action=RL.choose_action(observation)
            
            reward, observation_ =env.step(action)
                
            RL.store_transition(observation, action, reward, observation_)

            observation=observation_

            if (step > 200) and (step % 5 == 0):
                    RL.learn()

            if env.terminate():
                print("EPISODE: {}\ncurrent reward: {}\ncurrent state: {}".format(i,env.reward(),env.S))
                
                break
            step+=1
    print("end training")
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
    RL.plot_cost()
        
    