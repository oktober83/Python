import numpy as np
import matplotlib.pyplot as plt

x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
yb = [0.740, 0.785, 0.801, 0.805, 0.813, 0.809, 0.810, 0.826, 0.824, 0.824]
ya = [0.79, 0.839, 0.848, 0.862, 0.874, 0.873, 0.87, 0.874, 0.874, 0.878]
yab = [0.798, 0.84, 0.848, 0.863, 0.873, 0.873, 0.871, 0.877, 0.877, 0.879]

plt.plot(x, yb, 'ro-',markersize=8)
# plt.title('Effectiveness of Basic Naive Bayes Classifier')
plt.title('Effectiveness of Naive Bayes Classifier with Basic Features')
plt.xlabel('Percentage of Training Data used (%)')
plt.ylabel('Accuracy of Classifier on Validation Data')
plt.grid()

plt.show()



