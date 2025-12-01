from typing import Dict, Any, Union

class EnvConfig:
    def __init__(self, config: Dict[str, Any] = None): 
        self.step: int = 8
        self.num_of_epochs: int = 3
        self.w_s: float = 0.2 
        self.w_r: float = 0.8
        self.sigma: float = 1.5
        self.learning_rate: float = 0.7 
        self.activation_distance: str = 'euclidean'
        self.topology: str = 'circular'
        self.neighborhood_function: str = 'gaussian'
        self.random_seed: int = 10 
        if config: 
            for property, value in config.items():
                if hasattr(self, property):
                    setattr(self, property, value)
                else:
                    print(f"property {property} not found in EnvConfig")