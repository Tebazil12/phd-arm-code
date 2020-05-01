import scipy.io
import numpy as np


def load_data():
    n_angles = 19
    all_data = [None] * n_angles # all_data[angle][disp][frame][pin][xory]

    folder_path = "C:\\Users\\ea-stone\\Documents\\data\\singleRadius2019-01-16_1651\\"

    for i in range(0,n_angles):
        if i < 9:
            file_name = 'c0' + str(i+1) + '_01.mat'
        else:
            file_name = 'c' + str(i+1) + '_01.mat'

        full_load_name = folder_path + file_name
        mat = scipy.io.loadmat(full_load_name)
        all_data[i] = mat['data'][0]
        # all_data[i].round(2)

    return all_data


def get_ref_tap(all_data):
    ref_tap_all_frames = all_data[10 - 1][11 - 1]
    ref_tap_all_frames_norm = ref_tap_all_frames - ref_tap_all_frames[0]
    # print(len(ref_tap_all_frames_norm))
    # print(len(ref_tap_all_frames_norm[0]))
    # print(len(ref_tap_all_frames_norm[0][0]))
    # print(np.mean(ref_tap_all_frames_norm,axis=1))
    mean_all_frames = np.mean(ref_tap_all_frames_norm,axis=1)
    # print((np.mean(ref_tap_all_frames_norm, axis=1)).shape)
    # print(np.linalg.norm(mean_all_frames, axis=1))
    distances_all_disps = np.linalg.norm(mean_all_frames, axis=1)
    # print(distances_all_disps.max())
    result = np.where(distances_all_disps == np.amax(distances_all_disps))
    max_frame_i = result[0][0]
    ref_tap = ref_tap_all_frames_norm[max_frame_i]

    # print(np.sqrt(mean_all_frames[:,0]**2 + mean_all_frames[:,1]**2))

    # print(np.max(ref_tap_all_frames_norm))
    return ref_tap


def get_training_data(all_data):
    sigma_n_diss = 5
    i_training_angles = [10-1, 15-1, 19-1, 5-1, 1-1]

    for i_angle in range(0, len(i_training_angles)):

        [dissims,y_train] = process_taps(all_data[i_training_angles[i_angle]])



    return False


def process_taps(radii_data):
    # radii_data[disp][frame][pin][xory]
    print(len(radii_data))

    for disp_num in range(0,len(radii_data)):
        tap_norm = radii_data[disp_num] - radii_data[disp_num][1]
        print(tap_norm)

    return [False,False]



if __name__ == "__main__":
    all_data = load_data()
    print(len(all_data[0]))

    ref_tap = get_ref_tap(all_data)
    # print(ref_tap)
    # print((ref_tap.shape))

    a = get_training_data(all_data)
