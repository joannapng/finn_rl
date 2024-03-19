import numpy as np
import matplotlib.pyplot as plt

accuracy = np.array([99.45, 99.45, 98.792, 97.55, 9.45])
size = np.array([1.0, 1.0, 0.525, 0.45, 0.125])

plt.figure()
plt.plot(-size, accuracy, marker = '.', ms = 20, mfc = 'r')
plt.title('Accuracy vs Size (LeNe5)')
plt.xlabel('- Size of model (% of largest quantized model)')
plt.ylabel('Accuracy (MNIST)')
plt.savefig('size_vs_accuracy.png')
