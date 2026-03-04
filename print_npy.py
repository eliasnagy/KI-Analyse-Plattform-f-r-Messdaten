"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)
arr = np.load("numpy_files/c1_features_X.npy", allow_pickle=False)
print("shape:", arr.shape)
print("dtype:", arr.dtype)
print("erste 5 Zeilen:\n", arr[:5])
"""


import numpy as np
cols = ['Force_X','Force_Y','Force_Z','Vibration_X','Vibration_Y','Vibration_Z','AE_RMS']
stats = ['mean','std','max','min','rms']
feat_names = [f"{c}_{s}" for c in cols for s in stats]
arr = np.load("numpy_files/c1_features_X.npy")
for name, val in zip(feat_names, arr[0]):
    print(f"{name}: {val}")

X = np.load("numpy_files/c1_features_X.npy"); print(X.shape)
y = np.load("numpy_files/c1_features_y.npy"); print(y.shape)
print("erste y:", y[:5])