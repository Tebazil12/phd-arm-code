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


def best_frame(all_frames):
    """
    For the given tap, select frame with the highest average pin displacement from the first frame.
    Takes in one tap, in form (16ish x 126 ish x 2) np.array, returns (1 x 126ish x 2) np.array
    """
    # make into pin displacement instead of absolute position
    all_frames_disp = all_frames - all_frames[0]

    # Find frame where largest pin displacement takes place (on average) #TODO this method is not the same as in MATLAB!
    # Average disps per frame (over all pins)
    mean_disp_per_frame = np.mean(np.abs(all_frames_disp),axis=1)

    # Find euclidean distance per frame
    distances_all_disps = np.linalg.norm(mean_disp_per_frame, axis=1)

    # Find frame with max euclidean distance
    result = np.where(distances_all_disps == np.amax(distances_all_disps))
    max_frame_i = result[0][0]

    return all_frames_disp[max_frame_i]


def get_training_data(all_data):
    sigma_n_diss = 5
    i_training_angles = [10-1, 15-1, 19-1, 5-1, 1-1]

    for i_angle in range(0, len(i_training_angles)):

        [dissims,y_train] = process_taps(all_data[i_training_angles[i_angle]])



    return False


def process_taps(radii_data):
    # radii_data[disp][frame][pin][xory]
    # print(len(radii_data))

    for disp_num in range(0,len(radii_data)):
        tap_disps = radii_data[disp_num] - radii_data[disp_num][1]
        # print(tap_disps)

    return [False,False]



if __name__ == "__main__":
    all_data = load_data()
    # print(len(all_data[0]))

    ref_tap = best_frame(all_data[10 - 1][11 - 1]) # -1 to translate properly from matlab
    # print(ref_tap)
    print((ref_tap.shape))

    a = get_training_data(all_data)
