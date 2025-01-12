from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

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