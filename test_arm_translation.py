import load_json
import numpy as np

def test_whatever():
    ref_diffs_norm = load_json.whatever()
    assert ref_diffs_norm != None

def test_load_data():
    all_data = load_json.load_data()

    # Check all_data is correct format

    assert type(all_data) is list
    # print(len(all_data))
    assert type(all_data[1]) is list
    # print(type(all_data[1]))
    # print(len(all_data[1]))
    assert type(all_data[1][1]) is list
    # print(type(all_data[1][1]))
    # print(len(all_data[1][1]))
    assert type(all_data[1][1][1]) is np.ndarray
    # print(type(all_data[1][1][1]))
    # print(len(all_data[1][1][1]))
    # print((all_data[1][1][1]))
    # print(type(all_data[1][1][1][1]))
    assert type(all_data[1][1][1][1]) is np.ndarray
    # print(type(all_data[1][1][1][1][1]))
    assert type(all_data[1][1][1][1][1]) is np.ndarray
    # print(type(all_data[1][1][1][1][1][1]))
    assert type(all_data[1][1][1][1][1][1]) is np.float64

    # all_data[1][1][1] = None

    # data[tap_number][frame][pin][xorydisp]

    for n_depths in all_data:
        for n_angles in n_depths:
            if any(elem is None for elem in n_angles):
                assert False