import numpy as np 

class CorridorEnv:
    def __init__(self, n_states=3, is_slippery = False):
        self.nS = n_states
        self.nA = 3
        self.is_slippery = is_slippery
        self.s = 0
        self.prob = 0.9
        self.terminal_s = self.nS-1        
        self.model = {}
        for s in range(self.nS):
            self.model[s] = {}
            for a in range(self.nA):
                self.s = s
                s_, dropped = self.step(a)
                p1 = self.prob
                p2 = 1 - self.prob
                r = -10 if dropped else -1 
                d = bool(s == self.terminal_s)
                self.model[s][a] = [(p1, s_, r, d), (p2, np.int64(s), r, d)] if self.is_slippery else [(1.0, s_, r, d)]
        
        
    def reset(self):
        self.s = 0
        
    def step(self, a):
        if a == 0: # LEFT
            shift = -1
        elif a == 1: # DON'T MOVE
            shift = 0
        elif a == 2:
            shift = +1 # RIGHT
        else:
            raise

        if self.is_slippery:
            self.s = np.random.choice([self.s+shift, self.s], p=[self.prob, 1 - self.prob])
        else:
            self.s += shift
            
        dropped = (self.s<0 or self.s>=self.nS)
        return np.clip(self.s, 0, self.nS-1), dropped

class CorridorEnvTimePenalty:
    def __init__(self, n_states=3, is_slippery = False):
        self.nS = n_states
        self.nA = 3
        self.is_slippery = is_slippery
        self.s = 0
        self.prob = 0.9
        self.terminal_s = self.nS-1        
        self.model = {}
        for s in range(self.nS):
            self.model[s] = {}
            for a in range(self.nA):
                self.s = s
                s_, dropped = self.step(a)
                p1 = self.prob
                p2 = 1 - self.prob
                done = bool(s == self.terminal_s)
                r = -10 if (dropped and not done) else -1    
                self.model[s][a] = [(p1, s_, r, done), (p2, np.int64(s), r, done)] if self.is_slippery else [(1.0, s_, r, done)]
        
        
    def reset(self):
        self.s = 0
        
    def step(self, s, a):
        if a == 0: # LEFT
            shift = -1
        elif a == 1: # DON'T MOVE
            shift = 0
        elif a == 2:
            shift = +1 # RIGHT
        else:
            raise

        if self.is_slippery:
            self.s = np.random.choice([s+shift, s], p=[self.prob, 1 - self.prob])
        else:
            self.s += shift
            
        dropped = (s<0 or s>=self.nS)
        return np.clip(s, 0, self.nS-1), dropped

class CorridorEnvTerminalReward:
    def __init__(self, n_states=3, is_slippery = False):
        self.nS = n_states
        self.nA = 3
        self.is_slippery = is_slippery
        self.s = 0
        self.prob = 0.9
        self.terminal_s = self.nS-1      
        self.model = {}
        for s in range(self.nS):
            self.model[s] = {}
            for a in range(self.nA):
                # Pre-calculate transition dynamics for the model
                self.model[s][a] = self.step(s, a)
        
    def reset(self):
        self.s = 0
        return self.s
        
    def step_environment(self, s, a):
        if a == 0: # LEFT
            shift = -1
        elif a == 1: # STAY
            shift = 0
        elif a == 2: # RIGHT
            shift = +1
        else:
            raise ValueError("Invalid action")

        # Logic for slippery transitions
        if self.is_slippery:
            # Note: This logic assumes if you slip, you stay in the 's' you were in
            next_s = np.random.choice([s + shift, s], p=[self.prob, 1 - self.prob])
        else:
            next_s = s + shift
            
        dropped = (next_s < 0 or next_s >= self.nS)
        return np.clip(next_s, 0, self.nS-1), dropped

    def step(self, s, a):
        s_, dropped = self.step_environment(s, a)
        
        done = bool(s == self.terminal_s)
        r = 10.0 if done else 0.0
        
        if self.is_slippery:
            # p1: success move to s_, p2: slip stays at current s
            # We must re-evaluate 'done' and 'r' for the slip case (p2)
            done_slip = bool(s == self.terminal_s)
            r_slip = 10.0 if done_slip else 0.0
            return [(self.prob, s_, r, done), (1 - self.prob, np.int64(s), r_slip, done_slip)]
        else:
            return [(1.0, s_, r, done)]
