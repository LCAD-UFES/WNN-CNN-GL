import os
import numpy as np
from dataset_util import *
import matplotlib.pyplot as plt


def base_datasetname(dir, year_base, year_curr, offset_base, offset_curr):
    return dir + "basepos-{0}-{1}-{2}m-{3}m.txt".format(year_base, year_curr, offset_base, offset_curr)


def curr_datasetname(dir, year_base, year_curr, offset_base, offset_curr):
    return dir + "livepos-{0}-{1}-{2}m-{3}m.txt".format(year_base, year_curr, offset_base, offset_curr)


def find_closest_in_space(curr_pose, base_poses, curr_time, base_times, base_offset):
    nearest_index = -1
    smallest_distance = base_offset
    shortest_interval = np.float('inf')
    for j in range(len(base_poses)):
        interval = np.abs(curr_time - base_times[j])
        distance = LA.norm(curr_pose[[0,1]]-base_poses[j][[0,1]])
        # remove pontos na contra-mao
        orientation = np.abs(curr_pose[2] - base_poses[j][2])
        if (orientation <= math.pi/2) \
                and (distance >= 0) and (distance <= smallest_distance) \
                and (interval >= 0) and (interval <= shortest_interval):
            smallest_distance = distance
            shortest_interval = interval
            nearest_index = j
    return nearest_index


def detect_closure_loop(data, min_distance):
    index = -1
    first = np.array((data['x'][0], data['y'][0]))
    previous_distance = 0
    for i in range(1, len(data)):
        current = np.array((data['x'][i], data['y'][i]))
        current_distance = LA.norm(first - current)
        if (previous_distance - current_distance) > 0.1 and current_distance <= min_distance:
            index = i
        previous_distance = current_distance
    return index


def plot_dataset(data1, data2, labels1, labels2):
    plt.figure(figsize=(10, 6), dpi=100)
    data1_scatter = plt.scatter(data1[:,0], data1[:,1], facecolors='g', edgecolors='g', alpha=.5, s=5)
    data2_scatter = plt.scatter(data2[:,0], data2[:,1], facecolors='r', edgecolors='r', alpha=.5, s=5)
    for x2, l2 in enumerate(labels2):
        for x1, l1 in enumerate(labels1):
            if l1 == l2:
                plt.plot([data1[x1, 0], data2[x2, 0]], [data1[x1, 1], data2[x2, 1]])
                break
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend((data1_scatter, data2_scatter), ('Base', 'Live'), loc='upper left')
    plt.show()


def save_dataset_base(data, labels, dataset):
    sample_file = open(dataset, "w")
    sample_file.write("image label x y z rx ry rz timestamp\n")
    for i in range(len(data)):
        sample_file.write("{0} {1} {2} {3} {4} {5} {6} {7} {8}\n".format(
            data['left_image'][i], labels[i],
            data['x'][i], data['y'][i], data['z'][i],
            data['rx'][i], data['ry'][i], data['rz'][i],
            data['timestamp'][i])
        )
    sample_file.close()


def create_dataset(datasetname_label, datasetname_base, datasetname_curr, datasetname_base_out, datasetname_curr_out, offset_base, offset_curr):
    data_label = np.genfromtxt(datasetname_label, delimiter=' ', names=True, dtype=np.dtype(columns))
    data_base = np.genfromtxt(datasetname_base, delimiter=' ', names=True, dtype=np.dtype(columns))
    data_curr = np.genfromtxt(datasetname_curr, delimiter=' ', names=True, dtype=np.dtype(columns))

    data_label_loop = detect_closure_loop(data_label, offset_base)
    data_base_loop = detect_closure_loop(data_base, offset_base)
    data_curr_loop = detect_closure_loop(data_curr, offset_curr)

    data_label = data_label[:data_label_loop]
    data_base = data_base[:data_base_loop]
    data_curr = data_curr[:data_curr_loop]

    data_label_index = get_indices_of_sampled_data(data_label, offset_base)
    data_label = data_label[data_label_index]
    data_label_label = np.arange(data_label.shape[0])
    # data_label_label = np.random.permutation(data_label.shape[0])

    data_curr_index = get_indices_of_sampled_data(data_curr, offset_curr)
    data_curr = data_curr[data_curr_index]

    data_base_index = get_indices_of_sampled_data(data_base, offset_curr)  # initially sample at same rate as curr (assuming offset_curr <= offset_base)
    data_base = data_base[data_base_index]

    data_label_pose_2d = np.dstack([data_label['x'], data_label['y'], data_label['rz']])[0]  # x, y, yaw
    data_base_pose_2d = np.dstack([data_base['x'], data_base['y'], data_base['rz']])[0]  # x, y, yaw
    data_curr_pose_2d = np.dstack([data_curr['x'], data_curr['y'], data_curr['rz']])[0]  # x, y, yaw

    base_start, label_start = find_start_point(data_base_pose_2d, data_label_pose_2d)
    data_base_time = build_spacial_index(data_base_pose_2d, base_start)
    data_label_time = build_spacial_index(data_label_pose_2d, label_start)
    data_base_index = []
    data_base_label = []
    for index_base in range(len(data_base)):
        index_label = find_closest_in_space(data_base_pose_2d[index_base], data_label_pose_2d,
                                           data_base_time[index_base], data_label_time, 5.0)
        if index_label < 0:
            continue

        data_base_label.append(data_label_label[index_label])
        data_base_index.append(index_base)

    curr_start, label_start = find_start_point(data_curr_pose_2d, data_label_pose_2d)
    data_curr_time = build_spacial_index(data_curr_pose_2d, curr_start)
    data_label_time = build_spacial_index(data_label_pose_2d, label_start)
    data_curr_index = []
    data_curr_label = []
    for index_curr in range(len(data_curr)):
        index_label = find_closest_in_space(data_curr_pose_2d[index_curr], data_label_pose_2d,
                                           data_curr_time[index_curr], data_label_time, 5.0)
        if index_label < 0:
            continue

        data_curr_label.append(data_label_label[index_label])
        data_curr_index.append(index_curr)

    # find centroids of clusters in data_curr_pose_2d
    last_label = data_curr_label[0]
    sum_centroid = data_curr_pose_2d[data_curr_index[0]]
    num_centroid = 1
    centroids = {}
    for index in range(1, len(data_curr_label)):
        if data_curr_label[index] == last_label:
            coord = data_curr_pose_2d[data_curr_index[index]]
            sum_centroid = sum_centroid + coord
            num_centroid = num_centroid + 1
        else:
            centroids[last_label] = sum_centroid / num_centroid
            last_label = data_curr_label[index]
            sum_centroid = data_curr_pose_2d[data_curr_index[index]]
            num_centroid = 1
    centroids[last_label] = sum_centroid / num_centroid

    # find closest data_base_pose_2d to each centroid
    data_base_label2 = []
    data_base_index2 = []
    nearest_index = -1
    smallest_distance = np.float('inf')
    last_label = data_base_label[0]
    for index in range(1, len(data_base_label)):
        index_base = data_base_index[index]
        if data_base_label[index] in centroids:
            if data_base_label[index] == last_label:
                distance = LA.norm(centroids[data_base_label[index]][[0, 1]] - data_base_pose_2d[index_base][[0, 1]])
                if distance < smallest_distance:
                    smallest_distance = distance
                    nearest_index = index_base
            else:
                if nearest_index >= 0:
                    data_base_label2.append(last_label)
                    data_base_index2.append(nearest_index)
                nearest_index = -1
                smallest_distance = np.float('inf')
                last_label = data_base_label[index]
        else:
            print(data_base_label[index])
    data_base_index = data_base_index2

    data_base_label = np.array(data_base_label2)
    data_base = data_base[data_base_index]

    data_curr_label = np.array(data_curr_label)
    data_curr = data_curr[data_curr_index]

    save_dataset_base(data_base, data_base_label, datasetname_base_out)
    save_dataset_base(data_curr, data_curr_label, datasetname_curr_out)

    # data_curr_pose_2d = np.dstack([data_curr['x'], data_curr['y'], data_curr['rz']])[0]
    # data_base_pose_2d = np.dstack([data_base['x'], data_base['y'], data_base['rz']])[0]
    # plot_dataset(data_base_pose_2d, data_curr_pose_2d, data_base_label, data_curr_label)


if __name__ == '__main__':
    input_dir = '/dados/ufes/'
    # output_dir = '/home/avelino/deepslam/data/ufes_wnn/'
    # output_dir = '/Users/avelino/Sources/deepslam/data/ufes_wnn/'
    output_dir = '/home/likewise-open/LCAD/avelino/deepslam/data/ufes_wnn/'
    # offset_base_list = [1, 5, 10, 15, 30]
    offset_base_list = [5]
    offset_curr = 1

    # os.system('rm -rf ' + output_dir + '*')
    # datasets = ['20161021', '20171122']
    # terceira ponte (camera 3)
    # datasets = ['20160906-02', '20161228', '20170220', '20170220-02']
    # datasets = ['20160830', '20170119']
    # volta da ufes (camera 3)
    datasets = ['20160825', '20160825-01', '20160825-02', '20171205', '20180112', '20180112-02', '20171122']
    # volta da ufes (camera 8)
    # datasets += ['20140418', '20160902', '20160906-01']
    for k in range(0, len(offset_base_list)):
        offset_base = offset_base_list[k]
        for i in range(0, len(datasets)):  # base datasets
            for j in range(len(datasets)-1, len(datasets)):  # curr datasets
                # if i != j: continue  # skips building base and curr datasets with different data
                if i == j: continue  # skips building base and curr datasets with same data
                basefilename_in = logfilename(input_dir, datasets[i])
                currfilename_in = logfilename(input_dir, datasets[j])
                basefilename_out = base_datasetname(output_dir, datasets[i], datasets[j], offset_base, offset_curr)
                currfilename_out = curr_datasetname(output_dir, datasets[i], datasets[j], offset_base, offset_curr)
                labelfilename = logfilename(input_dir, '20140418')
                if not os.path.isfile(currfilename_out):
                    print 'building ', basefilename_out, currfilename_out
                    create_dataset(labelfilename, basefilename_in, currfilename_in, basefilename_out, currfilename_out, offset_base, offset_curr)
                else:
                    print 'skipping ', basefilename_out, currfilename_out

