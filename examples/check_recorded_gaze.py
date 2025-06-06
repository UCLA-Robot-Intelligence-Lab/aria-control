import numpy as np

npy_file = "../recordings/20250520_193447/raw_gaze.npy"

data = np.load(npy_file, allow_pickle=True)
print(f"Loaded array with shape {data.shape} and dtype {data.dtype}")

print("First 10 entries:")
print(data[:10])
