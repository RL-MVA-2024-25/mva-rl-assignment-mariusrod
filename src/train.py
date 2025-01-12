from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
#from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
#import matplotlib.pyplot as plt

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

env_rdm = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class ProjectAgent:
    def act(self, observation, use_random=False):
        SA =[]
        for i in range(4):
            SA.append(np.append(observation,i))
        SA = np.array(SA)
        return self.Q.predict(SA).argmax()

    def save(self, path):
        joblib.dump(self.Q, path)

    def load(self):
        self.Q = joblib.load("Random_forest_Q.pkl")
        print("Model loaded!")

class Trainer:
    def __init__(self, env_list, agent):
        self.env_list = env_list
        self.g_env = self._gen_env()
        self.env = next(self.g_env)

        self.agent = agent
        if not hasattr(self.agent, "Q"):
            self.agent.Q = RandomForestRegressor()
            self.Q_init = False
        else:
            self.Q_init = True

        self.fig = plt.figure()
        plt.ion()
        plt.show()

    def _gen_env(self):
        while True:
            for env in self.env_list:
                yield env
        
    def get_next_env(self):
        return next(self.g_env)

    def collect_samples(self, horizon, use_random_policy = True, disable_tqdm=False, print_done_states=False):
        env = self.env
        s, _ = env.reset()
        #dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(horizon), disable=disable_tqdm):
            if use_random_policy:
                a = env.action_space.sample()
            else:
                if np.random.rand()<0.1:
                    a = env.action_space.sample()
                else:
                    a = self.agent.act(s)
            s2, r, done, trunc, _ = env.step(a)
            #dataset.append((s,a,r,s2,done,trunc))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                env = self.get_next_env()
                s, _ = env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D

    def rf_fqi(self, S, A, R, S2, D, iterations, nb_actions, gamma, epsilon = 0.1, batch_size = 100, disable_tqdm=False):
        s, _ = self.env.reset()
        nb_samples = S.shape[0]
        SA = np.append(S,A,axis=1)
        mean_value = []
        keep_cond = np.ones(nb_samples, dtype=bool)
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            #random_batch = np.random.choice(nb_samples, batch_size, replace=False)
            #keep_cond = [i in random_batch for i in range(nb_samples)]
            if iter==0 and not self.Q_init:
                value=R[keep_cond].copy()
            else:
                Q2 = np.zeros((nb_samples,nb_actions)) #replace nb_samples by by batch_size if you want to use a batch
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((S[keep_cond].shape[0],1))
                    S2A2 = np.append(S2[keep_cond],A2,axis=1)
                    Q2[:,a2] = self.agent.Q.predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R[keep_cond] + gamma*(1-D[keep_cond])*max_Q2
            self.agent.Q = RandomForestRegressor()
            self.agent.Q.fit(SA[keep_cond],value)
            mean_value.append(value.mean()) 
            plt.clf()
            plt.plot(mean_value)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            if iter%10==0:
                self.agent.save("Random_forest_Q.pkl")


            # step in environment
            for n_step in range(200):
                if np.random.rand()<epsilon:
                    a = self.env.action_space.sample()
                    s2, r, d, t, _ = self.env.step(a)
                    #s, a, r, s2, d = self.collect_samples(self.env, 1, use_random_policy=True, disable_tqdm=True)
                else:
                    a = self.agent.act(s)
                    s2, r, d, t, _ = self.env.step(a)
                    #s, a, r, s2, d = self.collect_samples(self.env, 1, use_random_policy=False, disable_tqdm=True)
                S = np.append(S[1:],[s],axis=0)
                A = np.append(A[1:],[[a]],axis=0)
                R = np.append(R[1:],[r],axis=0)
                S2 = np.append(S2[1:],[s2],axis=0)
                D = np.append(D[1:],[d],axis=0)
                SA = np.append(S,A,axis=1)
                if d or t:
                    self.env = self.get_next_env()
                    s, _ = self.env.reset() # could be removed
                    break
                else:
                    s = s2
        return mean_value
            
    def train(self, horizon, iterations, epochs, nb_actions, gamma):
        reward = []
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            if epoch==0:
                use_random_policy = True # set to false if checkpoints are loaded
            else:
                use_random_policy = False
            S, A, R, S2, D = self.collect_samples(horizon, use_random_policy=use_random_policy)
            mean_value = self.rf_fqi(S, A, R, S2, D, iterations, nb_actions, gamma)
            reward.append(mean_value)
        plt.ioff()
        plt.figure(2)
        plt.plot(np.array(reward).flatten())
        plt.xlabel("Iterations")
        plt.ylabel("Mean value")
        plt.title("Mean value over iterations")
        plt.show()

if __name__=="__main__":
    horizon = 20000
    iterations = 2000
    epochs = 1
    nb_actions = 4
    gamma = 0.99

    agent = ProjectAgent()
    #agent.load() #uncomment to load a checkpoint
    trainer = Trainer([env_rdm, env], agent)
    trainer.train(horizon, iterations, epochs, nb_actions, gamma)
    agent.save("Random_forest_Q.pkl")
    print("Training done!")

        

