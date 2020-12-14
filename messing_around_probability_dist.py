import numpy as np

probs = [1.0] + [0.0]*9

choices = np.random.choice([i for i in range(10)], 500, p=probs, replace=True)

print(choices)
