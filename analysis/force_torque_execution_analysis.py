import cPickle as pickle
import numpy as np
import os
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.cm as cm
from matplotlib import animation
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
import subprocess
import cv_bridge
bridge = cv_bridge.CvBridge()

np.set_printoptions(precision=2, suppress=True)

ROOT = '/home/main/data/2021_visual_servoing/initial_tests'
FORCE_COLS = ['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ']
label_map = {0: 'No Contact', 1: 'Contact', 2: 'Failure', 3: 'Success'}

def extract_label(ts, label_set):
    label = 0
    for next_ts, next_label in label_set:
        if next_ts > ts:
            break
        label = next_label
    return label

def message_to_array(msg):
    force = msg.wrench.force
    torque = msg.wrench.torque
    array = [force.x, force.y, force.z, torque.x, torque.y, torque.z]
    return np.array(array)


def process_data(data, subsample=500, file_id=None):
    labels = data['labels']
    wrenches = data['/wrench']
    stamps = sorted(wrenches)

    if subsample is not None and len(wrenches) > subsample:
        choice = np.random.choice(len(wrenches), subsample, replace=False)
        wrenches = {stamps[i]: wrenches[stamps[i]] for i in choice}

    all_observations = []
    for ts, msg in wrenches.iteritems():
        label = extract_label(ts, labels)
        all_observations.append(np.concatenate([message_to_array(msg), [ts, label, file_id]]))

    return np.array(all_observations)


def svd_by_class_analysis(data):
    for label, subdata in data.groupby('Label'):
        vals = subdata[FORCE_COLS].values
        if len(vals) > 2000:
            vals = vals[np.random.choice(len(vals), 2000, replace=False)]
        u, s, v = np.linalg.svd(vals, compute_uv=True)
        print('\n' + '=' * 30 + '\nCLASS {}\n'.format(label_map[label]))
        print('Components:')
        print(v)
        print('Values:')
        print(s)


def plot_heatmap(x_vals, y_vals, grid_size, sigma=100, title=None, range=None, xlabel=None, ylabel=None, normalize_cutoff=0.5, ax=None):

    # import pdb
    # pdb.set_trace()

    heatmap, xedges, yedges = np.histogram2d(x_vals, y_vals, range=range, bins=grid_size)
    heatmap = heatmap.T

    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    if normalize_cutoff:
        idx = heatmap > 0
        heatmap[idx] = heatmap[idx] + normalize_cutoff * (1 - heatmap[idx])


    heatmap = gaussian_filter(heatmap, sigma=sigma)



    cm_to_use = cm.jet

    if ax is not None:
        ax.imshow(heatmap, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', aspect='auto', cmap=cm_to_use)
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
    else:
        plt.imshow(heatmap, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', aspect='auto', cmap=cm_to_use)
        # plt.imshow(heatmap, origin='lower', cmap=cm.jet)
        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.show()


def neural_network_analysis(dataset):
    dataset = dataset.sort_values('Time', ascending=True, inplace=False)
    dataset.reset_index(inplace=True, drop=True)

    data_folder = os.path.join(ROOT, 'network_testing')
    script_path = '/home/main/python/force_classification/force_classification.py'
    assert isinstance(dataset, pd.DataFrame)
    data_path = os.path.join(data_folder, 'data.csv')
    dataset.to_csv(data_path)

    subprocess.call(['python3', script_path, 'process', data_path])

    dataset = pd.DataFrame.from_csv(data_path)
    return dataset

def plot_file_predictions(file_path, file_name=None, title=None, ax=None):
    with open(file_path, 'rb') as fh:
        data = pickle.load(fh)

    if not file_path.endswith('.pickle'):
        raise NotImplementedError
    if file_name is None:
        plot_output = file_path.replace('.pickle', '.pdf')
    else:
        plot_output = os.path.join(os.path.dirname(file_path), file_name)

    cols = ['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ', 'Time', 'Label', 'Trial']
    force_data = pd.DataFrame(process_data(data, None, 0), columns=cols)
    force_data = force_data.sort_values('Time').reset_index(drop=True)

    min_time = force_data['Time'].min()
    max_time = force_data['Time'].max()
    camera_topic = '/camera/color/image_raw/compressed'

    force_data = neural_network_analysis(force_data)

    cols = ['FX', 'FY', 'FZ', 'p0', 'p1', 'p2', 'Label']

    interpolator = interp1d(force_data['Time'].values, force_data[cols].values, axis=0)

    all_index_ts = []
    all_header_ts = []

    for index_ts in sorted(data[camera_topic]):
        ts = data[camera_topic][index_ts].header.stamp.to_sec()
        if min_time <= ts <= max_time:
            all_index_ts.append(index_ts)
            all_header_ts.append(ts)

    all_interpolated_data = interpolator(all_header_ts)
    min_header = min(all_header_ts)
    all_header_ts = [x - min_header - 0.9 for x in all_header_ts]

    df = pd.DataFrame(all_interpolated_data, index=all_header_ts, columns=cols)
    # Hack to backfill
    df['Label'][df['Label'] % 1 > 0.001] = np.NaN
    df['Label'].fillna(method='ffill', inplace=True)
    df = df.dropna()
    df['Label'] = df['Label'].map({0:0, 1: 0, 2:1, 3:2})

    # Label grouping - Find bounds for filling
    bounds = []   # (left_bound, right_bound, label)
    start_idx = df.index[0]
    current_label = 0
    for idx in df.index:
        next_label = df.loc[idx, 'Label']
        if next_label != current_label:
            bounds.append((start_idx, idx, current_label))
            start_idx = idx
            current_label = next_label
    bounds.append((start_idx, df.index[-1], current_label))

    df['Magnitude'] = np.linalg.norm(df[['FX', 'FY', 'FZ']].values, axis=1)

    info_map = {
        'p0': {'title': 'Uncertain', 'color': 'darkgray'},
        'p1': {'title': 'Failure', 'color': 'firebrick'},
        'p2': {'title': 'Success', 'color': 'forestgreen'}
    }

    if ax is not None:
        for col in ['p0', 'p1', 'p2']:
            info = info_map[col]
            ax.plot(df.index, df[col], color=info['color'], label=info['title'], linewidth=2)

        # y_min, y_max = plt.ylim()
        ax.set_ylim(0, 1)
        ax.set_xlim(df.index[0], df.index[-1])
        for x1, x2, label in bounds:
            if label == 0:
                continue
            color = 'forestgreen' if label == 2 else 'firebrick'
            ax.fill_between([x1, x2], y1=0, y2=1, color=color, alpha=0.15)
        ax.legend(loc=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Confidence')
        ax.legend(['Uncertain', 'Failure', 'Success'])
        if title is not None:
            ax.set_title(title)
        return ax
    else:
        plt.figure(figsize=(6, 3.0), dpi=150)
        for col in ['p0', 'p1', 'p2']:
            info = info_map[col]
            plt.plot(df.index, df[col], color=info['color'], label=info['title'], linewidth=2)

        # y_min, y_max = plt.ylim()
        plt.ylim(0,1)
        plt.xlim(df.index[0], df.index[-1])
        for x1, x2, label in bounds:
            if label == 0:
                continue
            color = 'forestgreen' if label == 2 else 'firebrick'
            plt.fill_between([x1, x2], y1=0, y2=1, color=color, alpha=0.15)
        plt.legend(loc=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Confidence')
        plt.legend(['Uncertain', 'Failure', 'Success'])
        if title is not None:
            plt.title(title)

        plt.show()




def create_video_from_file(file_path, file_name=None):
    with open(file_path, 'rb') as fh:
        data = pickle.load(fh)

    if not file_path.endswith('.pickle'):
        raise NotImplementedError
    if file_name is None:
        video_output = file_path.replace('.pickle', '.mp4')
    else:
        video_output = os.path.join(os.path.dirname(file_path), file_name)

    cols = ['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ', 'Time', 'Label', 'Trial']
    force_data = pd.DataFrame(process_data(data, None, 0), columns=cols)
    force_data = force_data.sort_values('Time').reset_index(drop=True)

    min_time = force_data['Time'].min()
    max_time = force_data['Time'].max()
    camera_topic = '/camera/color/image_raw/compressed'

    force_data = neural_network_analysis(force_data)

    cols = ['FY', 'FZ', 'p0', 'p1', 'p2']

    interpolator = interp1d(force_data['Time'].values, force_data[cols].values, axis=0)

    all_index_ts = []
    all_header_ts = []

    for index_ts in sorted(data[camera_topic]):
        ts = data[camera_topic][index_ts].header.stamp.to_sec()
        if min_time <= ts <= max_time:
            all_index_ts.append(index_ts)
            all_header_ts.append(ts)

    all_interpolated_data = interpolator(all_header_ts)

    fig = plt.figure(figsize=(10.0, 8.0))
    force_ax = fig.add_subplot(211)
    img_ax = fig.add_subplot(223)
    class_ax = fig.add_subplot(224)

    # Prepare the image axis to have a data object to modify
    base_img = bridge.compressed_imgmsg_to_cv2(data[camera_topic][all_index_ts[0]], desired_encoding='rgb8')
    img_obj = img_ax.imshow(base_img)

    force_ax.set_title('Force')
    force_ax.set_xlabel('FY (Up-Down)')
    force_ax.set_ylabel('FZ (In-Out)')
    force_ax.set_xlim(-15, 15)
    # force_ax.set_xlim(force_data['FY'].min(), force_data['FY'].max())
    force_ax.set_ylim(-12, 2)
    # force_ax.set_ylim(force_data['FZ'].min(), force_data['FZ'].max())
    line = force_ax.plot([0], [0])[0]


    class_ax.set_title('Classification')
    class_ax.set_xlabel('Category')
    class_ax.set_ylabel('Confidence')
    class_ax.set_ylim(0, 1)
    class_ax.set_xlim(-0.5, 2.5)
    class_ax.set_xticks([0,1,2])
    class_ax.set_xticklabels(['Unsure', 'Failure', 'Success'])
    bars = class_ax.bar([0,1,2], [0,0,0])

    plt.ion()
    plt.show()

    def animate(i):
        if i < 0:
            i = 0
        if i >= len(all_index_ts):
            i = len(all_index_ts) - 1

        # Update image
        img_msg = data[camera_topic][all_index_ts[i]]
        new_img = bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding='rgb8')
        img_obj.set_data(new_img)

        # Update bars
        preds = all_interpolated_data[i,2:5]
        pred = np.argmax(preds)
        for rect, new_h in zip(bars, preds):
            rect.set_height(new_h)



        # Update plot lines
        force_vals = all_interpolated_data[:i+1,:2]
        line.set_xdata(force_vals[:,0])
        line.set_ydata(force_vals[:,1])

        if pred == 0:
            color = 'grey'
        else:
            base = preds[[1,2]].sum()
            color = np.array([preds[1]/base, preds[2]/base, 0])
            color[color > 1] = 1
            color[color < 0] = 0

        line.set_color(color)
        line_width = 1.0
        if pred != 0:
            # Adjust the line width based on how much the prediction dominates the other ones
            if pred == 1:
                dom = pred - np.max(preds[[0,2]])
            else:
                dom = pred - np.max(preds[[0,1]])
            line_width += dom * 4.0
        line.set_linewidth(line_width)

    interval = (all_index_ts[-1] - all_index_ts[0]) / len(all_index_ts) * 1000
    ani = animation.FuncAnimation(fig, animate, frames=range(len(all_index_ts) + 50),
                                  interval=interval)

    ani.save(video_output)
    print('Saved file to: {}'.format(video_output))

    fig.clf()

if __name__ == '__main__':

    ALL_FILES = sorted([f for f in os.listdir(ROOT) if f.endswith('.pickle')])
    CACHE_FILE = 'processed_data.pickle'



    # VIDEO ANALYSIS
    TEST_DATASETS = [1, 3, 7, 15, 21, 22, 31, 32, 58, 64, 76, 86]
    for trial_id in TEST_DATASETS:
        print('Processing Video {}...'.format(trial_id))
        create_video_from_file(os.path.join(ROOT, ALL_FILES[trial_id]), file_name='{}.mp4'.format(trial_id))
    raise Exception


    try:
        with open(os.path.join(ROOT, CACHE_FILE), 'rb') as fh:
            data = pickle.load(fh)
            data = data.sort_values('Time')
    except IOError:
        print('Processing all data...')
        all_data = []
        for file_id, file in enumerate(ALL_FILES):
            with open(os.path.join(ROOT, file), 'rb') as fh:
                data = pickle.load(fh)
            if not data['labels']:
                continue

            all_data.append(process_data(data, None, file_id))
        all_data = np.concatenate(all_data, axis=0)
        cols = ['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ', 'Time', 'Label', 'Trial']
        data = pd.DataFrame(all_data, columns=cols)
        data = data.sort_values('Time')
        print('All done! Saving to cache...')



        with open(os.path.join(ROOT, CACHE_FILE), 'wb') as fh:
            pickle.dump(data, fh)

    ft_data = data[['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ']].values

    # # OUTPUTTING PLOTS
    # TEST_DATASETS = [1, 3, 7, 15, 21, 22, 31, 32, 58, 64, 76, 86]
    # SUCCESSES = [21, 22, 32]
    # for trial_id in TEST_DATASETS:
    #     terminal_label = data[data['Trial'] == trial_id]['Label'].values[-1]
    #     if terminal_label == 0 or terminal_label == 1:
    #         continue
    #     print('TERMINAL LABEL FOR TRIAL {}: {}'.format(trial_id, terminal_label))
    #     if trial_id in SUCCESSES:
    #         title = 'Network Predictions (Success)'
    #     else:
    #         title = 'Network Predictions (Failure)'
    #     plot_file_predictions(os.path.join(ROOT, ALL_FILES[trial_id]), file_name='{}.pdf'.format(trial_id), title=title)

    # PLOT ONLY FOR EXAMPLES
    fig = plt.figure(figsize=(6.4, 4.2), dpi=150)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    cases = [15, 21]
    plot_file_predictions(os.path.join(ROOT, ALL_FILES[cases[0]]), title='Network Predictions (Failure)', ax=ax1)
    plot_file_predictions(os.path.join(ROOT, ALL_FILES[cases[1]]), title='Network Predictions (Success)', ax=ax2)

    ax1.set_xlabel('')

    plt.show(fig)

    # # Neural network analysis test
    # for _, subdata in data.groupby('Trial'):
    #     neural_network_analysis(subdata)
    #
    #
    #
    #
    # # # SVDs by class
    # svd_by_class_analysis(data)
    #
    #
    # # =========================
    # # PLOT PROGRESSION
    # # =========================
    #
    # # Figure out what the terminal category of a run is, and plot all runs
    #
    #
    #
    # TIME_SPLITS = 8
    #
    # y_bounds = (data['FY'].min(), data['FY'].max())
    # z_bounds = (data['FZ'].min(), data['FZ'].max())
    # for label_of_interest in [3, 2]:
    #
    #     ax = plt.axes()
    #     ax.set_xlabel('FY (Up-Down)')
    #     ax.set_ylabel('FZ (In-Out)')
    #     ax.set_title('Progression for {} Trials'.format(label_map[label_of_interest]))
    #     ax.set_xlim(*y_bounds)
    #     ax.set_ylim(*z_bounds)
    #
    #
    #     for i, (trial, subdata) in enumerate(data.groupby('Trial')):
    #
    #         if label_of_interest == 2 and i > 20:
    #             break
    #
    #         subdata = subdata.sort_values('Time')
    #
    #         terminal_value = subdata['Label'].values[-1]
    #         if terminal_value != label_of_interest:
    #             continue
    #
    #         subdata['Time'] = subdata['Time'] - subdata['Time'].min()
    #         subdata['Alpha'] = np.floor((0.9999 * subdata['Time'] / subdata['Time'].max()) * TIME_SPLITS) / TIME_SPLITS
    #
    #         color = np.random.uniform(size=3)
    #         color = (color / np.linalg.norm(color)) * 0.5
    #
    #         for (alpha, label), subsubdata in subdata.groupby(['Alpha', 'Label']):
    #             ls = '-' if label == label_of_interest else ':'
    #             ax.plot(subsubdata['FY'], subsubdata['FZ'], color=color, alpha=alpha, linestyle=ls)
    #
    #     plt.show()










    # ==========================
    # HEATMAPS
    # ==========================

    min_fy = data['FY'].min()
    max_fy = data['FY'].max()
    min_fz = data['FZ'].min()
    max_fz = data['FZ'].max()

    sigma=2.0

    # noncontact_pts = data[data['Label'] == 0]
    # contact_pts = data[data['Label'] != 0]
    #
    # plot_heatmap(noncontact_pts['FY'], noncontact_pts['FZ'], 500, range=[[min_fy, max_fy], [min_fz, max_fz]], sigma=sigma,
    #              title='Non-Contact State Forces', xlabel='FY (Up-Down)', ylabel='FZ (In-Out)')
    # plot_heatmap(contact_pts['FY'], contact_pts['FZ'], 500, range=[[min_fy, max_fy], [min_fz, max_fz]], sigma=sigma,
    #              title='Contact State Forces', xlabel='FY (Up-Down)', ylabel='FZ (In-Out)')


    success_pts = data[data['Label'] != 0]
    # success_pts = data[data['Label'] == 3]
    failure_pts = data[data['Label'] == 0]
    # failure_pts = data[data['Label'] == 2]

    fig = plt.figure(figsize=(6.4, 3.2), dpi=150)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    plot_heatmap(success_pts['FY'], success_pts['FZ'], 500, range=[[min_fy, max_fy], [min_fz, max_fz]],
                 sigma=sigma,
                 title='Contact State Forces', xlabel='FY (Up-Down) (N)', ylabel='FZ (In-Out) (N)', ax=ax1)

    plot_heatmap(failure_pts['FY'], failure_pts['FZ'], 500, range=[[min_fy, max_fy], [min_fz, max_fz]],
                 sigma=sigma,
                 title='Non-Contact State Forces', xlabel='FY (Up-Down) (N)', ax=ax2)
    ax2.yaxis.set_visible(False)

    plt.show(fig)

    import pdb
    pdb.set_trace()


    u, s, v = np.linalg.svd(ft_data, compute_uv=True)



    # FORCE PLOTS
    ax = plt.axes(projection='3d')

    subsample_pts = 250
    label_map = {0: 'No Contact', 1: 'Contact', 2: 'Failure', 3: 'Success'}
    markers = {0: 'o', 1: '^', 2: 'x', 3: '*'}
    colors = {0: 'blue', 1: 'orange', 2: 'red', 3: 'green'}
    for label_val, sub_df in data.groupby('Label'):
        to_retrieve = sub_df.values[np.random.choice(len(sub_df), min(len(sub_df), subsample_pts), replace=False)]
        ax.scatter3D(to_retrieve[:,0], to_retrieve[:,1], to_retrieve[:,2], label=label_map[label_val], marker=markers[label_val], color=colors[label_val])

    ax.set_title('Force')
    ax.set_xlabel('X (Left-Right)')
    ax.set_ylabel('Y (Up-Down)')
    ax.set_zlabel('Z (In-Out)')

    plt.legend()
    plt.show()

    # TORQUE PLOTS
    ax = plt.axes(projection='3d')

    subsample_pts = 250

    markers = {0: 'o', 1: '^', 2: 'x', 3: '*'}
    for label_val, sub_df in data.groupby('Label'):
        to_retrieve = sub_df.values[np.random.choice(len(sub_df), min(len(sub_df), subsample_pts), replace=False)]
        ax.scatter3D(to_retrieve[:, 3], to_retrieve[:, 4], to_retrieve[:, 5], label=label_map[label_val], color=colors[label_val],
                     marker=markers[label_val])
    ax.set_title('Torque')
    ax.set_xlabel('X (Left-Right)')
    ax.set_ylabel('Y (Up-Down)')
    ax.set_zlabel('Z (In-Out)')

    plt.legend()
    plt.show()
