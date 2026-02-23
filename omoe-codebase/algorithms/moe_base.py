import numpy as np
from itertools import combinations
from math import prod, sqrt, log
from collections import defaultdict
import math
from random import random, choice, randint
from config.variables import *

class PseudoExpert:
    def __init__(self, p, id=None):
        self.id = id
        self.successes = 0
        self.trials = 0
        self.p = p  # True success probability
    
    def update(self, success):
        self.successes += int(success)
        self.trials += 1
        self.p_est = self.successes / max(1, self.trials)
    
    def get_estimate(self):
        return self.successes / max(1, self.trials)
    
    def get_lcb(self, alpha_learn=ALPHA_LEARN):
        """Lower confidence bound"""
        if self.trials == 0:
            return -float('inf')
        # delta_t = 1/self.trials
        raw_cb = self.get_estimate() - alpha_learn*math.sqrt(math.log(self.trials) / self.trials)
        lcb = np.clip(raw_cb, 0, 1)
        return lcb
    
    def get_ucb(self, alpha_learn=ALPHA_LEARN):
        """Upper confidence bound"""
        if self.trials == 0:
            return float('inf')
        # delta_t = 1/self.trials
        raw_cb = self.get_estimate() + alpha_learn*math.sqrt(math.log(self.trials) / self.trials)
        ucb = np.clip(raw_cb, 0, 1)
        return ucb

    def get_lcb_delta(self, delta=DELTA_CONFIDENCE):
        """Lower confidence bound"""
        if self.trials == 0:
            return -float('inf')
        return self.get_estimate() - math.sqrt(math.log(1/delta) / (2 * self.trials))
    
    def get_ucb_delta(self, delta=DELTA_CONFIDENCE):
        """Upper confidence bound"""
        if self.trials == 0:
            return float('inf')
        return self.get_estimate() + math.sqrt(math.log(1/delta) / (2 * self.trials))
    
    def play(self):
        return random() < self.p