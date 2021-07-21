#!/usr/bin/env python

import rospy
from nmea_msgs.msg import Sentence
from sensor_msgs.msg import CompressedImage, PointCloud2, Image
from ros_numpy import numpify
import os
import cv2
import numpy as np
import sys
import rosbag
import struct
from contextlib import contextmanager
from cv_bridge import CvBridge
from copy import deepcopy
# from gps_mapping_time_synchronizer import write_ply_from_pc
import pandas as pd
import ros_numpy.point_cloud2 as pc2

def process_gps_string(msg):
    comps = msg.split(',')
    lat = convert_to_fractional_degrees(*comps[2:4])
    long = convert_to_fractional_degrees(*comps[4:6])
    return lat, long

def convert_to_fractional_degrees(degree_msg, dir):

    if dir in ['S', 'W']:
        mult = -1
    else:
        mult = 1

    idx = degree_msg.index('.')
    degs = int(degree_msg[0:idx-2])
    mins = float(degree_msg[2:])
    return mult * (degs + mins / 60)

def load_gps_info():
    bag_folder = '/home/main/data/data_collection_oct2020'
    output_file = os.path.join(bag_folder, 'gps_timestamps.csv')
    if os.path.exists(output_file):
        print('Loading GPS data from cache...')
        df = pd.read_csv(output_file)
        if 'Time' in df.columns:
            df = df.set_index('Time')
        return df


    # all_files = [x for x in os.listdir(bag_folder) if x.endswith('.bag') and 'converted' not in x]
    all_files = [x for x in os.listdir(bag_folder) if x.endswith('_converted.bag')]
    # file_map = {int(x.replace('.bag', '').split('_')[-1]): x for x in all_files}
    file_map = {int(x.replace('_converted.bag', '').split('_')[-1]): x for x in all_files}

    all_rez = []
    for num in sorted(file_map):
        file = file_map[num]
        bag_file = os.path.join(bag_folder, file)
        bag = rosbag.Bag(bag_file)

        for topic, msg, _ in bag.read_messages():
            if topic != '/gps_message':
                continue
            t = msg.header.stamp.to_sec()
            lat, long = process_gps_string(msg.sentence)

            all_rez.append([t, lat, long])

    if not all_rez:
        raise Exception('No entries processed?')

    all_rez = pd.DataFrame(all_rez, columns=['Time', 'Lat', 'Long'])
    all_rez = all_rez.set_index('Time')
    all_rez.to_csv(output_file)
    return all_rez

def interpolate_df_row(df, t):
    vals = df.index.values
    truth = vals > t
    if np.all(truth) or np.all(~truth):

        raise Exception("Cannot interpolate a time outside of the DataFrame's time frame! (Got {:.4f}, range is ({:.4f}, {:.4f})".format(t, np.min(vals), np.max(vals)))

    latter_idx = np.argmax(truth)
    t_latter = df.index[latter_idx]
    t_former = df.index[latter_idx - 1]

    scale = (t - t_former) / (t_latter - t_former)
    interp = df.iloc[latter_idx - 1] + scale * (df.iloc[latter_idx] - df.iloc[latter_idx - 1])
    return interp.reindex(['Lat', 'Long'])






if __name__ == '__main__':

    bag_folder = '/home/main/data/data_collection_oct2020/throttled_data'
    output_folder = '/home/main/data/data_collection_oct2020/throttled_data/processed_data'

    df = load_gps_info()

    import pdb
    pdb.set_trace()

    bag_files = sorted([x for x in os.listdir(bag_folder) if x.endswith('.bag')])

    all_rez = []
    for file in bag_files:
        print(file)
        bag_path = os.path.join(bag_folder, file)
        bag = rosbag.Bag(bag_path)

        for topic, msg, _ in bag.read_messages():
            if type(msg).__name__.endswith('PointCloud2'):
                stamp = msg.header.stamp
                nsec = stamp.to_nsec()
                sec = stamp.to_sec()
                output_path = os.path.join(output_folder, '{}.ply'.format(nsec))
                if not os.path.exists(output_path):
                    # Annoying hack
                    if not isinstance(msg, PointCloud2):
                        slots = ['header', 'height', 'width', 'fields', 'is_bigendian', 'point_step', 'row_step', 'data','is_dense']
                        old_msg = msg
                        msg = PointCloud2()
                        for slot in slots:
                            msg.__setattr__(slot, old_msg.__getattribute__(slot))

                    pts = numpify(msg)
                    pts = pc2.split_rgb_field(pts).reshape(-1)
                    write_ply_from_pc(pts, output_path)

                lat, long = interpolate_df_row(df, sec).values
                all_rez.append([nsec, lat, long])

    df = pd.DataFrame(all_rez, columns=['nsec', 'Lat', 'Long'])
    df.to_csv(os.path.join(output_folder, 'gps_coordinates.csv'))









    # df = load_gps_info()
    # if 'Time' in df.columns:
    #     df = df.set_index('Time')
    # rand = np.random.uniform(np.min(df.index.values), np.max(df.index.values))
    # interpolated = interpolate_df_row(df, rand).reindex(['Lat', 'Long'])
    # print(interpolated)
