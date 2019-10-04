import json
import numpy as np
with open('/home/lizzie/OneDrive/data/collect_data_3d_varyAngle_FIXEDslice2019-10-01_1901/c45_01_20.json') as f:
    data = json.load(f)
# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
#print(data)
print(type(data))
print(len(data))

# data[tap_number][frame][pin][xorydisp]
n_disps = 21
n_angles = 19
n_depths = 9
n_radii = n_angles * n_depths

all_data= np.empty([n_depths, n_angles, n_disps], dtype = object)

current_number = 0
for depth in range(0,n_depths):
    for angle in range(0,n_angles):
        for disp in range(0,n_disps):
            # make list of arrays
            all_data[depth][angle][disp]= np.array(data[current_number])
            current_number = current_number + 1

# Check all_data is correct format
#print(len(all_data))
#print(len(all_data[1]))
#print(len(all_data[1][1]))
#print(len(all_data[1][1][1]))
#print((all_data[1][1][1]))
#print(type(all_data[1][1][1]))

x_real= np.empty([n_radii], dtype = object)
x_real[0] = np.arange(-10,11) #actually -10 to 10 but +1 cuz python
#print(x_real)

x_real_test = np.arange(-10,11) #actually -10 to 10 but +1 cuz python
X_SHIFT_ON = False
dissims =[]
