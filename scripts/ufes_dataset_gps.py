import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy import spatial
from numpy.random import RandomState
import cv2

prng = RandomState(123)

columns = [('x', float), ('y', float), ('z', float), ('rx', float), ('ry', float), ('rz', float),
           ('timestamp', object), ('left_image', object), ('right_image', object)]

def logfilename(year):
    pathname = "/dados/ufes"
    return pathname + "/camerapos-{0}.txt".format(year)

def datasetname(year, offset, mode):
    pathname = "/dados/ufes"
    return pathname + "/UFES-{0}-{1}-{2}.csv".format(year, offset, mode)

def savedataset_full(data, labels, year, offset, name="test"):
    sample_file = open(datasetname(year, offset, name), "w")
    sample_file.write("timestamp,x,y,label\n")
    for i in range(len(data)):
        sample_file.write("{0},{1},{2},{3}\n".format(data['timestamp'][i], data['x'][i], data['y'][i], labels[i]))
    sample_file.close()
    
def savedataset_partial_test(data, matches, year, offset, name="train"):
    sample_file = open(datasetname(year, offset, name), "w")
    sample_file.write("timestamp,x,y,label\n")
    for label, i in matches:
        sample_file.write("{0},{1},{2},{3}\n".format(data['timestamp'][i], data['x'][i], data['y'][i], label))
    sample_file.close()
    
def savedataset_partial_train(data, matches, year, offset, name="train"):
    sample_file = open(datasetname(year, offset, name), "w")
    sample_file.write("timestamp,x,y,label\n")
    for i, label in matches:
        sample_file.write("{0},{1},{2},{3}\n".format(data['timestamp'][i], data['x'][i], data['y'][i], label))
    sample_file.close()
    
def sample_data_indices(data, offset):
    last = None
    indices = []
    for i in range(len(data)):
        current = np.array( (data['x'][i], data['y'][i]) )
        
        if last is None:
            distance = offset
        else:
            distance = LA.norm(last - current)
        if (distance >= offset):
            indices.append(i)
            last = current
            
    return indices

def nearest_neighbors_A_from_B(A, B, metric='euclidean'):
    dAB = spatial.distance.cdist(A, B, metric=metric) #dAB is a matrix with size A rows and size B cols
    along_with_B = 1 #return the minimum values and corresponding indices of B
    return np.min(dAB, axis=along_with_B), np.argmin(dAB, axis=along_with_B)

def remove_duplicates_from_test(coords, labels, distances, threshold):
    path = []
    closest_sample = 0
    train_sample = labels[0]
    min_distance = float("inf")
    for sample in xrange(len(coords)):
        if labels[sample] == train_sample:
            curr_distance = distances[sample]
            if curr_distance <= min_distance:
                min_distance = curr_distance
                closest_sample = sample
        else:
            if min_distance <= threshold:
                path.append((train_sample, closest_sample))
            train_sample = labels[sample]
            closest_sample = sample
            min_distance = distances[sample]
    return path

def remove_duplicates_from_train(coords, labels, distances, threshold):
    path = []
    closest_sample = 0
    test_sample = labels[0]
    min_distance = float("inf")
    for sample in xrange(len(coords)):
        if labels[sample] == test_sample:
            curr_distance = distances[sample]
            if curr_distance < min_distance:
                min_distance = curr_distance
                closest_sample = sample
        else:
            if min_distance < threshold:
                path.append((closest_sample, test_sample))
            test_sample = labels[sample]
            closest_sample = sample
            min_distance = float("inf")
    return path

def plot_matches2d_taking_train_as_reference(data1, data2, labels):
    plt.figure(figsize=(10, 6), dpi=100)
    data1_scatter = plt.scatter(data1[:,0], data1[:,1], c='g', alpha=.4, s=5)
    data2_scatter = plt.scatter(data2[:,0], data2[:,1], c='r', alpha=.5, s=5)
    for x2, x1 in enumerate(labels):
        plt.plot([data1[x1,0], data2[x2,0]], [data1[x1,1], data2[x2,1]])

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend((data1_scatter, data2_scatter), ('Train', 'Test'), loc='upper left')
    plt.show()

def plot_matches2d_taking_test_as_reference(data1, data2, labels):
    plt.figure(figsize=(10, 6), dpi=100)
    data1_scatter = plt.scatter(data1[:,0], data1[:,1], c='g', alpha=.4, s=5)
    data2_scatter = plt.scatter(data2[:,0], data2[:,1], c='r', alpha=.5, s=5)
    for x1, x2 in enumerate(labels):
        plt.plot([data1[x1,0], data2[x2,0]], [data1[x1,1], data2[x2,1]])

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend((data1_scatter, data2_scatter), ('Train', 'Test'), loc='upper left')
    plt.show()

def plot_matches2d_taking_test_as_reference_resized(data1, data2, labels, data1_full, data2_full):
    plt.figure(figsize=(10, 6), dpi=100)
    #data1_mean = np.mean(data1_full, axis=0)
    #data1_full = (data1_full - data1_mean) * 2.5 + data1_mean
    #data1 = (data1 - data1_mean) * 2.5 + data1_mean
    data1_full = data1_full - np.array([-400, 200])
    data1 = data1 - np.array([-400, 200])
    plt.scatter(data1_full[:,0], data1_full[:,1], c='green', alpha=.1, s=2, marker='o', edgecolors='green')
    plt.scatter(data1[1:11,0], data1[1:11,1], c='green', alpha=1, s=50, marker='o')
    plt.scatter(data2_full[:,0], data2_full[:,1], c='red', alpha=.1, s=2, marker='o', edgecolors='red')
    plt.scatter(data2[1:4,0], data2[1:4,1], c='red', alpha=1, s=50, marker='o')
    for x1, x2 in enumerate(labels):
        if x2 > 0 and x2 < 4:
            plt.plot([data1[x1,0], data2[x2,0]], [data1[x1,1], data2[x2,1]])
    plt.axis("off")
    plt.show()

def show_dataset(year_pair, train, test, path):
    imagepath = "/dados/ufes"
    year_for_training = year_pair[0]
    year_for_testing  = year_pair[1]

    window = None
    plt.axis("off")
    for (sample_train, sample_test) in path:
        image_train = cv2.imread(imagepath + "/" + str(year_for_training) + "/" + train['timestamp'][sample_train] + '.bb08.l.png', cv2.IMREAD_COLOR)
        image_test = cv2.imread(imagepath + "/" + str(year_for_testing) + "/" + test['timestamp'][sample_test] + '.bb08.l.png', cv2.IMREAD_COLOR)
        image = np.hstack((image_train,image_test))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if window is None:
            window = plt.imshow(image)
        else:
            window.set_data(image)
        plt.pause(.1)
        plt.draw()

def generate_datasets_taking_train_as_reference(year_pair, offset_pair, threshold):
    print 'TRAIN DATASET AS REFERENCE:'
    year_for_training = year_pair[0]
    year_for_testing  = year_pair[1]
    offset_for_training = offset_pair[0]
    offset_for_testing  = offset_pair[1]
    
    train = np.genfromtxt(logfilename(year_for_training), delimiter=',', names=True, dtype=np.dtype([('timestamp', object), ('x', float), ('y', float)]))
    test = np.genfromtxt(logfilename(year_for_testing), delimiter=',', names=True, dtype=np.dtype([('timestamp', object), ('x', float), ('y', float)]))
    
    if year_for_training == 2012:
        train = train[65:train.shape[0]-160]
    if year_for_testing == 2014:
        test = test[140:-1]
    if year_for_training == 2014:
        train = train[140:-1]
    if year_for_testing == 2012:
        test = test[65:test.shape[0]-160]
    
    #coord_train_full = np.dstack([train['x'], train['y']])[0]
    coord_test_full = np.dstack([test['x'], test['y']])[0]
    
    indices = sample_data_indices(train, offset_for_training)
    train_data = train[indices]
    train_labels = np.arange(train_data.shape[0])
    
    coord_train = np.dstack([train_data['x'], train_data['y']])[0]
    
    distances, labels = nearest_neighbors_A_from_B(coord_test_full, coord_train)
    # many to one correspondences
    #correspondences = np.dstack([labels, np.arange(coord_test_full.shape[0])])[0]
    # one to one correspondences
    correspondences = remove_duplicates_from_test(coord_test_full, labels, distances, threshold)
    test = test[ np.array([index for label, index in correspondences]) ]
    labels = np.array([label for label, index in correspondences])
    indices = sample_data_indices(test, offset_for_testing)
    test_data = test[indices]
    test_labels = labels[indices]
    '''
    labels = np.array(labels)
    indices = sample_data_indices(test, offset_for_training) #this is ok! ratio will take care of the rest
    ratio = float(offset_for_training) / float(offset_for_testing)
    if ratio < 1.0:
        ratio = int(1.0 / ratio)
    else:
        ratio = int(ratio)
    test_data = test[indices[0:-1:ratio]]
    test_labels = labels[indices[0:-1:ratio]]
    '''
    
    savedataset_full(train_data, train_labels, year_for_training, offset_for_training, name='train')
    savedataset_full(test_data, test_labels, year_for_testing, offset_for_testing, name='test')
    
    print 'train dataset size after sampling by {0}m:'.format(offset_for_training), train_data.shape[0]
    print 'test dataset size after sampling by {0}m:'.format(offset_for_testing), test_data.shape[0]
    
    coord_test = np.dstack([test_data['x'], test_data['y']])[0]
    
    #plot_matches2d_taking_train_as_reference(coord_train, coord_test, test_labels)

def generate_datasets_taking_test_as_reference(year_pair, offset_pair, threshold):
    print 'TEST DATASET AS REFERENCE:'
    year_for_training = year_pair[0]
    year_for_testing  = year_pair[1]
    offset_for_training = offset_pair[0]
    offset_for_testing  = offset_pair[1]
    
    train = np.genfromtxt(logfilename(year_for_training), delimiter=' ', names=True, dtype=np.dtype(columns))
    test = np.genfromtxt(logfilename(year_for_testing), delimiter=' ', names=True, dtype=np.dtype(columns))
    '''
    if year_for_training == 2012:
        train = train[65:train.shape[0]-160]
    if year_for_testing == 2014:
        test = test[140:-1]
    if year_for_training == 2014:
        train = train[140:-1]
    if year_for_testing == 2012:
        test = test[65:test.shape[0]-160]
    '''
    coord_train_full = np.dstack([train['x'], train['y']])[0]
    #coord_test_full = np.dstack([test['x'], test['y']])[0]
    
    indices = sample_data_indices(test, offset_for_testing)
    test_data = test[indices]
    test_labels = np.arange(test_data.shape[0])
    
    coord_test = np.dstack([test_data['x'], test_data['y']])[0]
    
    distances, labels = nearest_neighbors_A_from_B(coord_train_full, coord_test)
    # many to one correspondences
    correspondences = np.dstack([np.arange(coord_train_full.shape[0]), labels])[0]
    # one to one correspondences
    # correspondences = remove_duplicates_from_train(coord_train_full, labels, distances, threshold)
    train = train[ np.array([index for index, label in correspondences]) ]
    labels = np.array([label for index, label in correspondences])
    indices = sample_data_indices(train, offset_for_training)
    train_data = train[indices]
    train_labels = labels[indices]
    '''
    labels = np.array(labels)
    indices = sample_data_indices(train, offset_for_testing) #this is ok! ratio will take care of the rest
    ratio = float(offset_for_training) / float(offset_for_testing)
    if ratio < 1.0:
        ratio = int(1.0 / ratio)
    else:
        ratio = int(ratio)
    train_data = train[indices[0:-1:ratio]]
    train_labels = labels[indices[0:-1:ratio]]
    '''

    savedataset_full(train_data, train_labels, year_for_training, offset_for_training, name='train')
    savedataset_full(test_data, test_labels, year_for_testing, offset_for_testing, name='test')
    
    print 'train dataset size after sampling by {0}m:'.format(offset_for_training), train_data.shape[0]
    print 'test dataset size after sampling by {0}m:'.format(offset_for_testing), test_data.shape[0]
    
    coord_train = np.dstack([train_data['x'], train_data['y']])[0]
    
    plot_matches2d_taking_test_as_reference(coord_train, coord_test, train_labels)
    
    #plot_matches2d_taking_test_as_reference_resized(coord_train, coord_test, train_labels, coord_train_full, coord_test_full)

def load_ufes_dataset_gps(imagepath, filename, shuffle_data=True, height=364, width=640):
    
    file_list = np.genfromtxt(filename, delimiter=',', names=True, dtype=np.dtype([('timestamp', object), ('x', float), ('y', float), ('label', int)]))

    samples = len(file_list)
    X = np.zeros((samples, 1, height/4, width/4), dtype=np.float32)
    y = np.zeros((samples, 1), dtype=np.int32)
    
    for sample in range(samples):
        image = cv2.imread(imagepath + file_list['timestamp'][sample] + '.bb08.l.png', cv2.IMREAD_GRAYSCALE) 
        image = image[0:height, 0:width]
        image = cv2.resize(image, (height/4, width/4), interpolation = cv2.INTER_AREA )
        image = image.reshape(1, height/4, width/4)
        
        X[sample] = image * (1./256.)
        y[sample] = file_list['label'][sample]
    
    X = X.astype(np.float32)
    #y = y.astype(np.float32)

    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(y.shape, y.min(), y.max()))

    if shuffle_data:
        shuffled_rows = prng.permutation(samples)
        X = X[shuffled_rows,:]
        y = y[shuffled_rows]

    return X, y

def compute_distance_velocity(data):
    distance_sum = 0
    velocity_mean = 0
    velocity_count = 0
    previous_pose = np.dstack((data['x'][0], data['y'][0]))[0][0]
    previous_time = data['timestamp'][0]
    for i in range(1, data.shape[0]):
        current_pose = np.dstack((data['x'][i], data['y'][i]))[0][0]
        current_time = data['timestamp'][i]
        distance = LA.norm(current_pose - previous_pose)
        velocity = distance/(current_time - previous_time)
        velocity_count = velocity_count + 1
        velocity_mean = velocity_mean + (velocity - velocity_mean)/velocity_count
        distance_sum = distance_sum + distance
        previous_pose = current_pose
        previous_time = current_time
    return velocity_mean, distance_sum

def generate_datasets_statistics(year_pair, threshold):
    year_for_training = year_pair[0]
    year_for_testing  = year_pair[1]
    
    train = np.genfromtxt(logfilename(year_for_training), delimiter=',', names=True, dtype=np.dtype([('timestamp', float), ('x', float), ('y', float)]))
    test = np.genfromtxt(logfilename(year_for_testing), delimiter=',', names=True, dtype=np.dtype([('timestamp', float), ('x', float), ('y', float)]))
    print 'DATASET STATISTICS:'
    
    print 'train dataset size before sampling:', train.shape[0]
    print 'test dataset size before sampling:', test.shape[0]
    
    print 'mean velocity and distance for train dataset:', compute_distance_velocity(train)
    print 'mean velocity and distance for test dataset:', compute_distance_velocity(test)
    
    coord_train = np.dstack([train['x'], train['y']])[0]
    coord_test = np.dstack([test['x'], test['y']])[0]
    
    distances, labels = nearest_neighbors_A_from_B(coord_train, coord_test)
    # many to one correspondences
    #correspondences = np.dstack([np.arange(coord_train.shape[0]), labels])[0]
    # one to one correspondences
    correspondences = remove_duplicates_from_train(coord_train, labels, distances, threshold)
    
    correspondences_count = 0
    correspondences_diff = np.zeros(len(correspondences))
    for index, label in correspondences:
        distance = LA.norm(coord_train[index] - coord_test[label])
        correspondences_diff[correspondences_count] = distance
        correspondences_count = correspondences_count + 1
        #print index, coord_train[index], coord_test[label], distance
        
    print "mean distance between matches:", np.mean(correspondences_diff), " and standard deviation: ", np.std(correspondences_diff)

if __name__ == '__main__':
    '''
    First experiment: benchmark ensemble with vgram using 2012/2014 for train/test keeping test fixed
    Second experiment: fine tunning the subsequence length using 2012/2014 for train/test keeping the same spacing
    Third experiment: measure the subsequence slope (step) impact using 2012/2014 for train/test keeping train fixed
    '''
    pairs_of_years_for_training_and_testing_datasets = [['20161021', '20171122']]
    offset_distance_in_meter_between_frames_same_spacing = [[1, 1]]  # [1, 1], [3, 3], [5, 5], [10, 10], [15, 15], [30, 30]]
    offset_distance_in_meter_between_frames_test_fixed = [[5, 1]]  # [[1, 1], [5, 1], [10, 1], [15, 1], [30, 1]]
    offset_distance_in_meter_between_frames_train_fixed = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]
    '''
    for year_pair in pairs_of_years_for_training_and_testing_datasets:
        generate_datasets_statistics(year_pair, threshold=15.0)
    '''
    for year_pair in pairs_of_years_for_training_and_testing_datasets:
        for offset_pair in offset_distance_in_meter_between_frames_test_fixed:
            generate_datasets_taking_test_as_reference(year_pair, offset_pair, threshold=15.0)
    '''
    for year_pair in pairs_of_years_for_training_and_testing_datasets:
        for offset_pair in offset_distance_in_meter_between_frames_train_fixed:
            generate_datasets_taking_train_as_reference(year_pair, offset_pair, threshold=15.0)
    for year_pair in pairs_of_years_for_training_and_testing_datasets:
        for offset_pair in offset_distance_in_meter_between_frames_test_fixed:
            generate_datasets_taking_test_as_reference(year_pair, offset_pair, threshold=15.0)
    '''
