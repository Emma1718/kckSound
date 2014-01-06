import numpy as np
import matplotlib.pyplot as plt

# generate some data
x = np.arange(0.,6.12,0.01)
y = np.sin(x)
# y = np.random.uniform(size=300)
yunbiased = y-np.mean(y)
ynorm = np.sum(yunbiased**2)
acor = np.correlate(yunbiased, yunbiased, "same")/ynorm
# use only second half
acor = acor[len(acor)/2:]

plt.plot(acor)
plt.show()
