import numpy as np
import matplotlib.pyplot as plt

'''
accuracy = np.array([99.45, 99.45, 98.792, 97.55, 9.45])
size = np.array([1.0, 1.0, 0.525, 0.45, 0.125])

plt.figure()
plt.plot(size, accuracy, marker = '.', ms = 20, mfc = 'r')
plt.title('Accuracy vs Size (LeNe5)')
plt.xlabel('Size of model (% of largest quantized model)')
plt.ylabel('Accuracy (MNIST)')
plt.savefig('size_vs_accuracy.png')
'''

import pandas as pd

weights = [[1.0, 0.0], [0.8, 0.2]] # do not use 0 because obviously 1-bit for the area part
#colors = ['g', 'b', 'r', 'm', 'k']
colors = ['g', 'b']
plt.figure()

for i, weight in enumerate(weights):
	agent = pd.read_csv(f'agent_{weights[i][0]}_{weights[i][1]}.monitor.csv', skiprows=0, header = 1)
	plt.plot(np.arange(np.shape(agent)[0]), agent['r'], c = colors[i], label = f'{weights[i][0]}_{weights[i][1]}')
plt.title('Reward for each agent per episode')
plt.legend()
plt.savefig('agents_monitor.png')
