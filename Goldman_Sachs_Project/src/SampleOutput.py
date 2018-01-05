import numpy as np

full = np.load('LinkMatrix.npy')
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=10000)


num_largest = 20

indices = (-full).argpartition(num_largest, axis=None)[:num_largest]
# OR, if you want to avoid the temporary array created by `-full`:
# indices = full.argpartition(full.size - num_largest, axis=None)[-num_largest:]

x, y = np.unravel_index(indices, full.shape)

print "x =", x
print "y =", y
print "Largest values:", full[x, y]
print "Compare to:    ", np.sort(full, axis=None)[-num_largest:]
print full[313][186]
