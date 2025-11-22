import numpy as np
from sklearn.neighbors import NearestNeighbors

class HybridPolicy:
    def __init__(self, bc_model, success_states_scaled, success_actions, dsafe=0.1, k=1):
        self.bc = bc_model
        self.states = success_states_scaled
        self.actions = success_actions
        self.nn = NearestNeighbors(n_neighbors=k).fit(success_states_scaled)

    def get_action(self, state_scaled):
        dist, idx = self.nn.kneighbors(state_scaled.reshape(1, -1))
        if dist[0][0] <= 0.1:
            return np.mean(self.actions[idx[0]], axis=0).tolist()
        else:
            return self.bc.predict(state_scaled.reshape(1, -1))[0].tolist()
