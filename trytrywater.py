# %%
import numpy as np
from tqdm import tqdm

# %%
p = np.zeros(shape=90)
p[:73] = 0.6 / 100
for i in range(73, 90):
    p[i] = p[i-1] + 6 / 100
p[p > 1.0] = 1.0

# %%
T = 100000
times = []
avg = 0.0
for i in tqdm(range(T), ncols=80):
    for j in range(90):
        if np.random.binomial(1, p=p[j]) == 1:
            avg += j + 1
            times.append(j+1)
            break

avg /= T
print('avg =', avg)

# %%
for j in range(90):
    print(np.random.binomial(1, p=p[j]))
