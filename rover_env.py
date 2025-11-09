# rover_env.py
import numpy as np
from cost_layers import make_synthetic_mars, CostLayers

class RoverEnv:
    """Synthetic environment fallback (not used for GeoTIFF path)."""
    ACTIONS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    def __init__(self, H=100, W=100, seed=42):
        self.H, self.W = H, W
        self.rng = np.random.default_rng(seed)
        self.layers = make_synthetic_mars(H, W, seed=seed)
        self.reset()

    def reset(self):
        free = np.argwhere(~self.layers.blocked)
        sidx = self.rng.integers(0, len(free))
        gidx = self.rng.integers(0, len(free))
        self.start = tuple(free[sidx])
        self.goal  = tuple(free[gidx])
        self.pos = self.start

class RoverEnvFromLayers:
    """Lightweight shim when you already built CostLayers from GeoTIFF."""
    ACTIONS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    def __init__(self, layers: CostLayers):
        self.layers = layers
        self.H, self.W = layers.slope.shape
        self.pos = (self.H//2, self.W//2)
        self.start = self.pos
        self.goal = (min(self.H-2, self.pos[0]+10), min(self.W-2, self.pos[1]+10))
