import scipy.io
import scipy.spatial
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    n_angles = 19
    all_data = [None] * n_angles  # all_data[angle][disp][frame][pin][xory]

    folder_path = "C:\\Users\\ea-stone\\Documents\\data\\singleRadius2019-01-16_1651\\"

    for i in range(0, n_angles):
        if i < 9:
            file_name = 'c0' + str(i + 1) + '_01.mat'
        else:
            file_name = 'c' + str(i + 1) + '_01.mat'

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
    mean_disp_per_frame = np.mean(np.abs(all_frames_disp), axis=1)

    # Find euclidean distance per frame
    distances_all_disps = np.linalg.norm(mean_disp_per_frame, axis=1)

    # Find frame with max euclidean distance
    result = np.where(distances_all_disps == np.amax(distances_all_disps))
    max_frame_i = result[0][0]

    tap = all_frames_disp[max_frame_i]

    return tap.reshape(tap.shape[0] * tap.shape[1])


def get_training_data(all_data, ref_tap):
    """Return two lists both the same length as number of training angles """

    sigma_n_diss = 5
    i_training_angles = [10 - 1, 15 - 1, 19 - 1, 5 - 1, 1 - 1]

    y_train_all = []
    dissim_all = []
    for i_angle in range(0, len(i_training_angles)):
        y_train_line = extract_ytrain(all_data[i_training_angles[i_angle]])
        dissim_line = calc_dissims(y_train_line, ref_tap)

        y_train_all.append(y_train_line)
        dissim_all.append(dissim_line)

    return [y_train_all, dissim_all]


def extract_ytrain(radii_data):
    """ Extract ytrain given radii_data[disp][frame][pin][xory] """

    # shape for the radius data to be returned (note this is 2d, not 3d)
    data_shape = (radii_data.shape[0], (radii_data[0][0].shape[0] * radii_data[0][0].shape[1]))
    y_train = np.zeros(shape=data_shape)

    for disp_num in range(0, len(radii_data)):  # for each tap on radius
        tap = best_frame(radii_data[disp_num])
        y_train[disp_num] = tap

    return y_train


def calc_dissims(y_train, ref_tap):
    # print("calc_dissim")

    diffs = -y_train + ref_tap
    # print(diffs.shape)

    # reshape to 21 by 126 by 2?
    diffs_3d = diffs.reshape(diffs.shape[0], int(diffs.shape[1]/2), 2)
    # print(diffs_3d.shape)

    y_train_2d = y_train.reshape(y_train.shape[0], int(y_train.shape[1]/2), 2)
    ref_tap_2d = ref_tap.reshape(int(ref_tap.shape[0] / 2), 2)

    sum_diffs = diffs_3d.sum(1) #sum in x and y
    # print(sum_diffs)
    # print(sum_diffs.shape)

    sum_ys = y_train_2d.sum(1)
    sum_ref = ref_tap_2d.sum(0)
    # print("ys and ref")
    print(sum_ys)

    print(ref_tap.shape)
    print(y_train.shape)
    # print(sum_ref)

    #TODO recreate matlab ordering of array(to see if this is causing the disparity):


    # Calculate Euclidean distance as dissimilarity measure
    dissim = np.linalg.norm(sum_diffs,axis=1)
    print("original dissim")
    print(dissim)

    # trying to recreate matlab - ignore, its the same results as the working python one, but hard to impolement
    # across rows properly in pythoon
    # dissim = np.linalg.norm(diffs_3d[1])
    # print("original dissim")
    # print(dissim)

    # dissim = scipy.spatial.distance.cdist(np.array([[0,0]]), sum_diffs, 'euclidean') #same as above method
    # print("dissim")
    # print(dissim)
    # print(dissim.shape)

    # dissim = scipy.spatial.distance.cdist([sum_ref], sum_ys, 'euclidean') #same as above 2 methods
    # print("dissim sums")
    # print(dissim[0])
    # print(dissim.shape)

    #todo this one works well
    # dissim = scipy.spatial.distance.cdist([ref_tap], y_train, 'euclidean') # NOTsame as above 2 methods
    # print(diffs.shape)

    #trying to replicate matlabs worse values
    # dissim = scipy.spatial.distance.cdist(np.empty(diffs.shape), diffs, 'euclidean')
    print("dissim sums")
    print(dissim)
    print(dissim.shape)

    # dissim = scipy.spatial.distance.cdist([ref_tap], y_train, 'cosine')

    # dissim = scipy.spatial.distance.cdist([sum_ref], sum_ys, 'cosine')
    # dissim = np.rad2deg(dissim)
    # print("dissim sums cosine")
    # print(dissim)
    # print(dissim.shape)
    # print(dissim_degs)

    return dissim[0] # so that not array within array...


def show_dissim_profile(x_real, dissim_all):
    print(dissim_all)
    for i in range(0, len(dissim_all)):
        plt.plot(x_real, dissim_all[i])
    ax = plt.gca()
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.ylabel('dissim')
    plt.xlabel('dissim')
    plt.show()

if __name__ == "__main__":
    all_data = load_data()
    # print(len(all_data[0]))

    ref_tap = best_frame(all_data[10 - 1][11 - 1])  # -1 to translate properly from matlab
    # print(ref_tap)
    # print((ref_tap.shape))

    [y_train_all, dissim_all] = get_training_data(all_data, ref_tap)

    # print(len(y_train_all))
    # print(y_train_all[0].shape)
    # print(len(dissim_all))
    # print(dissim_all[0].shape)
    x_real = -np.arange(-10,11)
    show_dissim_profile(x_real,dissim_all)

    #TODO calcultate line shift based of dissimilarity