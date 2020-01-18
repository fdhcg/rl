import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

__all__=['Env']

class Env():
    def __init__(self,p):
        self.p0=p
        self.p=None
        self.observation_space=10
        self.action_space=6


        self.S=np.random.rand(self.observation_space)
        self.hidden_S=1/100*np.random.rand()
        self.episode=0
        self.n_step=-1
        self.step_value=0.001
        self.maxlen=1000
        
        

    def take_action(self,n):
        A=np.array([0. if i !=n else self.step_value for i in range(self.action_space)])
        # self.S[0:int(self.action_space/2)]+=(A[:int(self.action_space/2)]-A[int(self.action_space/2):])
        A=(A[:int(self.action_space/2)]-A[int(self.action_space/2):])
        self.S[0:int(self.action_space/2)]+=A

        
        

    # def _env_h(self):
    #     step=np.zeros(1,dtype=int)
    #     def hidden_S_trans():
    #         self.hidden_S=self.hidden_S*np.exp(-step[0]*0.01)
    #         step[0]+=1
    #         print("step: {}".format(step[0]))
    #     return hidden_S_trans

    def _env_h(self):
        self.hidden_S=self.hidden_S*np.exp(-self.n_step*0.01)
        self.n_step+=1
        

    def _init(self,input):
        x=np.append(self.S,self.hidden_S)
        noise=(np.random.rand(5,2)-(1/2)*np.ones([5,2]))
        w=np.array([[0.02,0.04,0.09,0.01,0.04,0.02,0.04,0.06,0.03,0.01,1],
                    [0.01,0.02,0.03,0.02,0.05,0.02,0.09,0.10,0.02,0.04,1]])
        d=np.sum(x.dot(w.T),axis=-1)
        return input+1/1000*(noise+d)

        

    def _shift(self,input):
        x0=np.array([0.3,0.4])
        # x0=np.array([0.2,0.2,0.4,0.1,0.3])
        x=self.S[:2]
        # w=np.array([[0.3,0.4,0.9,0.1,0.8],[0.2,0.01,0.2,0.3,0.1]])
        # w=np.array([[0.3,0.4,0.9],[0.2,0.01,0.2]])
        # dx=np.sum((x[:2]-x0[:2])**2,axis=-1)
        # dy=np.sum((x[1:]-x0[1:])**2,axis=-1)
        dx=x[0]-x0[0]
        dy=x[1]-x0[1]
        input[:,0]+=dx
        input[:,1]+=dy
        return input


        
    def _rotate(self,input):
        x=self.S[:3]
        x0=np.array([0.3,0.4,0.5])
        theta=np.sum(((x-x0)**2),axis=-1)
        r=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
        return input.dot(r)

    def _zoom(self,input):
        x=self.S[::3]
        x0=np.array([0.2,0.1,0.7,0.4])
        w=np.array([0.7,0.6,0.1,0.4])
        h=np.array([[0.21,0.51],[0.52,0.4],[0.1,0.94],[1.22,0.92]])
        phi=np.sum((((x-x0)*w).dot(h))**2,axis=-1)
        return phi*input


    def step(self,A,verbose=False):
        if verbose:
            print("episode|step: {}|{}  take action #{}".format(self.episode,self.n_step,A))
        self.take_action(A)
        self._env_h()
        p=self._init(self.p0)
        # p=self._shift(p)
        p=self._rotate(p)
        # self.p=self._zoom(p)
        self.p=self._shift(p)

        return self.reward(),self.observation()

    def test(self):
        
        print(self._rotate(self.p))
        
        
    def terminate(self):
        if self.n_step>self.maxlen:
            return True
        elif np.max(self.S)>1 or np.min(self.S)<0:
            return True
        else:
            return False


    def observation(self):
        return self.output().reshape(-1)

    def output(self):
        return self.p-self.p0

    def plot_output(self,is_show=False,is_save=True):
        
        fig=plt.figure(figsize=(5, 5))
        ax=fig.add_subplot()
        for i in range(len(self.p)):
            ax.annotate('v'+str(i),xy=tuple(self.p[i]),xytext=tuple(self.p0[i]),arrowprops=dict(arrowstyle="->", color="r"))
        ax.grid()
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)
        ax.set_aspect('equal')
        if is_save:
            path="data/episode{}".format(self.episode)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path+"/step{}.png".format(self.n_step))
            if not is_show:
                plt.close()
            else:
                plt.ion()
                plt.draw()
                plt.pause(1)
                plt.close()
        



    # def reward(self):
    #     return 1/(np.sum(np.sum(self.output()**2)))
    def reward(self):
        l2=1/5*(np.sum(np.sum(self.output()**2)))
        if np.max(self.S)>1 or np.min(self.S)<0:
            return -100
        elif l2<0.01:
            return 10
        elif l2<0.05:
            return 5
        elif l2<0.1:
            return 2
        elif l2<1:
            return 1
        else:
            return 0


    def reset(self):
        self.S=np.random.rand(10)
        self.hidden_S=np.random.rand()
        self.episode+=1
        self.n_step=-1


if __name__ == "__main__":
    p=np.array([[1,1],[1,0],[0,1],[0,0],[1,-1]])
    E=Env(p)
    for i in range(100):
        R,S=E.step(1)
        print("R:{}\nS:{}".format(R,S))



