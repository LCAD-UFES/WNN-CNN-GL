import os
import numpy as np

columns = [('image', object), ('label', int),
           ('x', float), ('y', float), ('z', float),
           ('rx', float), ('ry', float), ('rz', float),
           ('timestamp', object)]

def base_datasetname(dir, year_base, year_curr, offset_base, offset_curr):
    return dir + "basepos-{0}-{1}-{2}m-{3}m.txt".format(year_base, year_curr, offset_base, offset_curr)


def curr_datasetname(dir, year_base, year_curr, offset_base, offset_curr):
    return dir + "livepos-{0}-{1}-{2}m-{3}m.txt".format(year_base, year_curr, offset_base, offset_curr)


def save_dataset(data, dataset):
    sample_file = open(dataset, "w")
    sample_file.write("image label x y z rx ry rz timestamp\n")
    for i in range(len(data)):
        sample_file.write("{0} {1} {2} {3} {4} {5} {6} {7} {8}\n".format(
            data['image'][i], data['label'][i],
            data['x'][i], data['y'][i], data['z'][i],
            data['rx'][i], data['ry'][i], data['rz'][i],
            data['timestamp'][i])
        )
    sample_file.close()


def concat_dataset(output_dir, trainfile, testfile, datasets, offset_base, offset_curr):
    datasetname_out = base_datasetname(output_dir, trainfile, testfile, offset_base, offset_curr)
    for i in range(0, len(datasets)):
        datasetname_in = base_datasetname(output_dir, datasets[i], testfile, offset_base, offset_curr)
        data_base = np.genfromtxt(datasetname_in, delimiter=' ', names=True, dtype=np.dtype(columns))
        if i == 0:
            data_all = data_base
        else:
            data_all = np.concatenate((data_all, data_base))
    data_all = np.sort(data_all, axis=0, order=['label', 'timestamp'])
    save_dataset(data_all, datasetname_out)

if __name__ == '__main__':
    input_dir = '/dados/ufes/'
    # output_dir = '/home/avelino/deepslam/data/ufes_wnn/'
    # output_dir = '/Users/avelino/Sources/deepslam/data/ufes_wnn/'
    output_dir = '/home/likewise-open/LCAD/avelino/deepslam/data/ufes_wnn/'
    offset_base = 5
    offset_curr = 1

    # os.system('rm -rf ' + output_dir + '*') 
    datasets = ['20160825', '20160825-01', '20160825-02', '20171205', '20180112', '20180112-02'] 
    dataset_test = '20171122'
    dataset_name = 'UFES-TRAIN-LAPS'
    concat_dataset(output_dir, dataset_name, dataset_test, datasets, offset_base, offset_curr)
    os.system('cp ' + curr_datasetname(output_dir, '20160825', dataset_test, offset_base, offset_curr) + ' ' + curr_datasetname(output_dir, dataset_name, dataset_test, offset_base, offset_curr)) 
