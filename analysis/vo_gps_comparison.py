import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d
import sys
python_version = sys.version_info.major
from matplotlib import animation
if python_version == 3:
    import utm
else:
    from nav_msgs.msg import Odometry
    import rosbag

DATA_ROOT = os.path.join('/home/main/data/data_collection_oct2020')

gps_file = os.path.join(DATA_ROOT, 'gps_timestamps.csv')
odom_bag_file = os.path.join(DATA_ROOT, 'odometry_readings.bag')
odom_file = os.path.join(DATA_ROOT, 'combined_data.csv')
def pose_to_array(pose):
    pos = pose.position
    rot = pose.orientation
    return np.array([pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, rot.w])

if __name__ == '__main__':
    try:
        df = pd.read_csv(odom_file)
    except IOError:

        if python_version != 2:
            raise Exception('Please run this mode in Python 2!')

        data = pd.read_csv(gps_file, index_col='Time')
        interpolator = interp1d(data.index, data[['Lat', 'Long']].values.T)
        bag_contents = rosbag.Bag(odom_bag_file)
        all_times = []
        all_poses = []

        for _, msg, _ in bag_contents.read_messages(topics=['/odom']):
            all_times.append(msg.header.stamp.to_sec())
            all_poses.append(pose_to_array(msg.pose.pose))

        all_data = pd.DataFrame(all_poses, index=all_times, columns=['x', 'y', 'z', 'rx', 'ry', 'rz', 'rw'])
        interped_gps = interpolator(all_times).T
        gps_df = pd.DataFrame(interped_gps, index=all_times, columns=['Lat', 'Long'])

        df = pd.concat([all_data, gps_df], axis=1)
        df.to_csv(odom_file)



    # WARNING! PAST HERE, NEED TO BE IN PYTHON 3
    if python_version != 3:
        raise Exception('Please run this section in Python 3!')
    # Turn GPS coords into XY coords
    if 'gps_x' not in df.columns:
        x_coords, y_coords, _, _ = utm.from_latlon(df['Lat'].values, df['Long'].values)
        coord_df = pd.DataFrame(np.array([x_coords, y_coords]).T, index=df.index, columns=['gps_x', 'gps_y'])
        df = pd.concat([df, coord_df], axis=1)
        df.to_csv(odom_file)

    if 'Unnamed: 0.1' in df.columns:
        df.set_index('Unnamed: 0.1', inplace=True)

    import matplotlib.pyplot as plt
    # Sample some number of GPS coordinates to use as the heading estimate (i.e. the transform of the base frame)



    SAMPLE_TIME = 5.0
    indexes = df.index[df.index - df.index[0] < SAMPLE_TIME]
    reg_vals = df.loc[indexes, ['gps_x', 'gps_y']].values
    reg_vals = reg_vals - reg_vals.mean(axis=0)

    v = np.linalg.svd(reg_vals, compute_uv=True)[2]
    comp = v[0]
    angle = -np.arctan2(comp[1], comp[0])

    rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    odom_tf = df[['x', 'y']].values.dot(rot_mat) + df.iloc[0][['gps_x', 'gps_y']].values

    # SETUP ANIMATION
    SPEEDUP = 3
    FPS = 20
    ANIM_FPS = SPEEDUP * FPS

    anim_times = np.arange(df.index[0], df.index[-1], 1/FPS)
    interp_gps = interp1d(df.index, df[['gps_x', 'gps_y']].values.T)(anim_times).T
    interp_odom = interp1d(df.index, odom_tf.T)(anim_times).T

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(min(min(interp_gps[:,0]), min(interp_odom[:,0])), max(max(interp_gps[:,0]), max(interp_odom[:,0])))
    ax.set_ylim(min(min(interp_gps[:,1]), min(interp_odom[:,1])), max(max(interp_gps[:,1]), max(interp_odom[:,1])))
    ax.set_aspect('equal')
    ax.legend(['GPS', 'Odom'], loc='lower right')

    line_gps = ax.plot([interp_gps[0, 0]], [interp_gps[0, 1]])[0]
    line_odom = ax.plot([interp_odom[0, 0]], [interp_odom[0, 1]])[0]

    plt.ion()
    plt.show()

    def animate(i):
        if i < 0:
            i = 0
        elif i >= len(anim_times):
            i = len(anim_times) - 1

        line_gps.set_xdata(interp_gps[:i+1,0])
        line_gps.set_ydata(interp_gps[:i+1,1])

        line_odom.set_xdata(interp_odom[:i+1,0])
        line_odom.set_ydata(interp_odom[:i+1,1])

        ax.set_title('{:.2f} (x{:.1f})'.format(anim_times[i] - anim_times[0], SPEEDUP))



    ani = animation.FuncAnimation(fig, animate, frames=range(len(anim_times)+50),
                                  interval=1000.0/ANIM_FPS)
    ani.save('gps_odom_comparison.avi')


    #
    #
    # plt.plot(df['gps_x'], df['gps_y'], label='GPS')
    # plt.plot(odom_tf[:,0], odom_tf[:,1], label='ICP-based Odometry')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend()
    # plt.show()







    # Transform all /odom frame into /base_link frame