import numpy as np
import matplotlib.pyplot as plt

kwng_error_log = np.load('kwng_error.npy')
sgd_error_log = np.load('sgd.npy')
plt.plot(range(len(kwng_error_log)), (kwng_error_log))
plt.plot(range(len(sgd_error_log)), (sgd_error_log))
plt.legend(["KWNG", "SGD"])
plt.ylabel("Training Error (log)")
plt.xlabel("EPOCH")
plt.savefig("loss-kwng-epoch.png", dpi=600, bbox_inches="tight")