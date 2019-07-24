import os
import math
from transform import *
from dataset_util import *
import matplotlib.pyplot as plt


def datasetname(dir, year, offset):
    return dir + "camerapos-{0}-{1}m.txt".format(year, offset)


def delta_datasetname(dir, year_base, year_live, offset_base, offset_live):
    return dir + "deltapos-{0}-{1}-{2}m-{3}m.txt".format(year_base, year_live, offset_base, offset_live)


def find_closest_in_space(curr_pose, base_poses, curr_time, base_times, base_offset):
    nearest_index = -1
    smallest_distance = 10.0
    shortest_interval = 10.0
    for j in range(len(base_poses)):
        if False:  # just frames ahead keyframe
            interval = curr_time - base_times[j]
            direction = signed_direction(curr_pose, base_poses[j])
            distance = direction * LA.norm(curr_pose[[0,1]]-base_poses[j][[0,1]])
        else: # get frames ahead and behind keyframe
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


def plot_dataset(data1, data2, matches):
    plt.figure(figsize=(10, 6), dpi=100)
    data1_scatter = plt.scatter(data1[:,0], data1[:,1], facecolors='g', edgecolors='g', alpha=.5, s=5)
    data2_scatter = plt.scatter(data2[:,0], data2[:,1], facecolors='r', edgecolors='r', alpha=.5, s=5)
    for x1, x2 in matches:
        plt.plot([data1[x1,0], data2[x2,0]], [data1[x1,1], data2[x2,1]])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend((data1_scatter, data2_scatter), ('Base', 'Live'), loc='upper left')
    plt.show()


def create_dataset(datasetname_base, datasetname_curr, datasetname_out, offset_base, cam_base, cam_curr, dataset_different):
    camera_frame = cam2tr()
    data_matches = []

    data_base = np.genfromtxt(datasetname_base, delimiter=' ', names=True, dtype=np.dtype(columns))
    data_curr = np.genfromtxt(datasetname_curr, delimiter=' ', names=True, dtype=np.dtype(columns))

    data_base_pose_2d = np.dstack([data_base['x'], data_base['y'], data_base['rz']])[0]  # x, y, yaw
    data_curr_pose_2d = np.dstack([data_curr['x'], data_curr['y'], data_curr['rz']])[0]  # x, y, yaw

    if dataset_different:
        curr_start, base_start = find_start_point(data_curr_pose_2d, data_base_pose_2d)
    else:
        curr_start, base_start = (0, 0)

    data_curr_index = build_spacial_index(data_curr_pose_2d, curr_start)
    data_base_index = build_spacial_index(data_base_pose_2d, base_start)

    outfile = open(datasetname_out, 'w')
    outfile.write('dx dy dz dr dp dy bx by bz br bp by lx ly lz lr lp ly')
    outfile.write(' base_depth base_image_left base_image_right')
    outfile.write(' live_depth live_image_left live_image_right')
    outfile.write(' base_fx base_cx base_fy base_cy base_baseline')
    outfile.write(' live_fx live_cx live_fy live_cy live_baseline timestamp\n')
    for index_curr in range(len(data_curr)):
        if offset_base > 0:
            index_base = find_closest_in_space(data_curr_pose_2d[index_curr], data_base_pose_2d,
                                            data_curr_index[index_curr], data_base_index,
                                            offset_base)
        else:
            index_base = index_curr-1

        if index_base < 0:
            continue

        base_position = np.array((data_base['x'][index_base], data_base['y'][index_base], data_base['z'][index_base])).reshape((3,1))
        base_rotation = rpy2r(data_base['rx'][index_base], data_base['ry'][index_base], data_base['rz'][index_base])
        base_pose = rt2tr(base_rotation, base_position)
        
        curr_position = np.array((data_curr['x'][index_curr], data_curr['y'][index_curr], data_curr['z'][index_curr])).reshape((3,1))
        curr_rotation = rpy2r(data_curr['rx'][index_curr], data_curr['ry'][index_curr], data_curr['rz'][index_curr])
        curr_pose = rt2tr(curr_rotation, curr_position)
        
        # compute displacement from frame base T-1 to curr T (i.e. forwards)
        delta_pose = np.linalg.inv(base_pose) * curr_pose
        # compute displacement from frame curr T to base T-1 (i.e. backwards)
        # delta_pose = np.linalg.inv(curr_pose) * base_pose
        
        # transform displacement to camera coordinate frame
        delta_pose = camera_frame * delta_pose * np.linalg.inv(camera_frame)
        delta_position = np.squeeze(np.asarray(tr2t(delta_pose)))
        delta_rotation = np.squeeze(np.asarray(tr2rpy(delta_pose)))
        
        if dataset_different and LA.norm(delta_position) < offset_curr:
            continue

        if LA.norm(delta_position) > offset_base > 0:
            continue

        data_matches.append((index_base, index_curr))
        
        # comment out snippet below to check transforms (forwards direction)
        '''
        # transform displacement back to carmen coordinate frame
        delta_pose = np.linalg.inv(camera_frame) * delta_pose * camera_frame
        # add displacement delta to base frame T-1 taking it back to curr T
        curr_pose_ = base_pose * delta_pose
        curr_position_ = np.squeeze(np.asarray(tr2t(curr_pose_)))
        curr_rotation_ = np.squeeze(np.asarray(tr2rpy(curr_pose_)))
        if np.allclose(curr_pose_, curr_pose)==False:
            print('Error')
        '''
        
        # comment out snippet below to check transforms (backwards direction)
        '''
        # transform displacement back to carmen coordinate frame
        delta_pose = np.linalg.inv(camera_frame) * delta_pose * camera_frame
        # add displacement delta to base frame T-1 taking it back to curr T
        base_pose_ = curr_pose * delta_pose
        base_position_ = np.squeeze(np.asarray(tr2t(base_pose_)))
        base_rotation_ = np.squeeze(np.asarray(tr2rpy(base_pose_)))
        if np.allclose(base_pose_, base_pose)==False:
            print('Error')
        '''
        
        outfile.write('{0:.6f}'.format(delta_position[0]) + ' ' +
                      '{0:.6f}'.format(delta_position[1]) + ' ' +
                      '{0:.6f}'.format(delta_position[2]) + ' ' +
                      '{0:.6f}'.format(delta_rotation[0]) + ' ' +
                      '{0:.6f}'.format(delta_rotation[1]) + ' ' +
                      '{0:.6f}'.format(delta_rotation[2]) + ' ')
        outfile.write('{0:.6f}'.format(data_base['x'][index_base]) + ' ' +
                      '{0:.6f}'.format(data_base['y'][index_base]) + ' ' +
                      '{0:.6f}'.format(data_base['z'][index_base]) + ' ' +
                      '{0:.6f}'.format(data_base['rx'][index_base]) + ' ' +
                      '{0:.6f}'.format(data_base['ry'][index_base]) + ' ' +
                      '{0:.6f}'.format(data_base['rz'][index_base]) + ' ')
        outfile.write('{0:.6f}'.format(data_curr['x'][index_curr]) + ' ' +
                      '{0:.6f}'.format(data_curr['y'][index_curr]) + ' ' +
                      '{0:.6f}'.format(data_curr['z'][index_curr]) + ' ' +
                      '{0:.6f}'.format(data_curr['rx'][index_curr]) + ' ' +
                      '{0:.6f}'.format(data_curr['ry'][index_curr]) + ' ' +
                      '{0:.6f}'.format(data_curr['rz'][index_curr]) + ' ')
        outfile.write(str(data_base['left_image' ][index_base]).replace('l.png','d.png') + ' ')
        outfile.write(str(data_base['left_image' ][index_base]) + ' ')
        outfile.write(str(data_base['right_image'][index_base]) + ' ')
        outfile.write(str(data_curr['left_image' ][index_curr]).replace('l.png','d.png') + ' ')
        outfile.write(str(data_curr['left_image' ][index_curr]) + ' ')
        outfile.write(str(data_curr['right_image'][index_curr]) + ' ')
        outfile.write('{0:.6f}'.format(cam_base['fx']) + ' ' +
                      '{0:.6f}'.format(cam_base['cx']) + ' ' +
                      '{0:.6f}'.format(cam_base['fy']) + ' ' +
                      '{0:.6f}'.format(cam_base['cy']) + ' ' +
                      '{0:.6f}'.format(cam_base['baseline']) + ' ')
        outfile.write('{0:.6f}'.format(cam_curr['fx']) + ' ' +
                      '{0:.6f}'.format(cam_curr['cx']) + ' ' +
                      '{0:.6f}'.format(cam_curr['fy']) + ' ' +
                      '{0:.6f}'.format(cam_curr['cy']) + ' ' +
                      '{0:.6f}'.format(cam_curr['baseline']) + ' ')
        outfile.write(str(data_curr['timestamp'][index_curr]))
        outfile.write('\n')
    outfile.close()
    # plot_dataset(data_base_pose_2d, data_curr_pose_2d, data_matches)


if __name__ == '__main__':
    input_dir = '/dados/ufes/'
    output_dir = '/home/avelino/deepslam/data/ufes_cnn/'
    # output_dir = '/Users/avelino/Sources/deepslam/data/ufes_cnn/'
    offset_base_list = [5]
    offset_curr = 1
    offset_head = math.pi/18.0
    camera3 = {'fx':0.764749,'cx':0.505423,'fy':1.01966,'cy':0.493814,'baseline':0.240040}
    camera8 = {'fx':0.753883,'cx':0.500662,'fy':1.00518,'cy':0.506046,'baseline':0.240031}

    # os.system('rm -rf '+output_dir+'*')
    
    # terceira ponte (camera 3)
    # datasets = ['20160906-02', '20161228', '20170220', '20170220-02']
    datasets = ['20160830' , '20170119'] # VALIDATION
    # volta da ufes (camera 3)
    datasets += ['20160825', '20160825-01', '20160825-02', '20161021', '20171205', '20180112', '20180112-02']
    datasets += ['20171122'] # TEST
    cameras = [camera3] * len(datasets)
    # volta da ufes (camera 8)
    # datasets += ['20140418', '20160902', '20160906-01']
    # cameras += [camera8] * 3
    for k in range(0, len(offset_base_list)):
        offset_base = offset_base_list[k]
        for i in range(0, len(datasets)):  # base datasets
            for j in range(0, len(datasets)):  # live datasets
                # if i != j: continue  # skips building base and live datasets with different data
                # if i == j: continue  # skips building base and live datasets with same data
                basefilename_in = logfilename(input_dir, datasets[i])
                basefilename_out = datasetname(output_dir, datasets[i], offset_base)
                currfilename_in = logfilename(input_dir, datasets[j])
                currfilename_out = datasetname(output_dir, datasets[j], offset_curr)
                datafilename_out = delta_datasetname(output_dir, datasets[i], datasets[j], offset_base, offset_curr)
                sample_dataset(basefilename_in, basefilename_out, offset_base, offset_head)
                sample_dataset(currfilename_in, currfilename_out, offset_curr, offset_head)
                if not os.path.isfile(datafilename_out):
                    print 'building ', datafilename_out
                    create_dataset(basefilename_out, currfilename_out, datafilename_out, offset_base, cameras[i], cameras[j], i != j)
                else:
                    print 'skipping ', datafilename_out
