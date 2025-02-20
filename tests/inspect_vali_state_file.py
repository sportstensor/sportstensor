import numpy as np

state_file = ''

# Load the .npz file
npz_file = np.load(state_file, allow_pickle=True)

# List the arrays contained in the file
print(npz_file.files)

print("step:", npz_file['step'])
print("scores:", npz_file['scores'])
print("uids_to_leagues:", npz_file['uids_to_leagues'])
print("uids_to_last_leagues:", npz_file['uids_to_last_leagues'])
print("uids_to_leagues_last_updated:", npz_file['uids_to_leagues_last_updated'])

# Access a specific array
#array_1 = npz_file['array_name_1']
#array_2 = npz_file['array_name_2']

# Use the arrays
#print(array_1.shape)
#print(array_2.dtype)

# Close the file
npz_file.close()