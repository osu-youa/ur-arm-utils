import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def process_gps_string(msg):
    """
    Takes in an NMEA GPS Sentence (as a standard Python string) and returns the
    corresponding latitude and longitude value.
    """

    comps = msg.split(',')
    lat = convert_to_fractional_degrees(*comps[2:4])
    long = convert_to_fractional_degrees(*comps[4:6])
    return lat, long

def convert_to_fractional_degrees(degree_msg, dir):
    """
    Takes in a degree value (expressed as a string) and a direction (NESW) and
    returns the corresponding value as a float.
    """

    if dir in ['S', 'W']:
        mult = -1
    else:
        mult = 1

    # The minutes consists of the two digits before the ., followed by all decimals after
    # Therefore we figure out where the . is and extract the corresponding degrees and minutes
    idx = degree_msg.index('.')
    degs = int(degree_msg[0:idx-2])
    mins = float(degree_msg[2:])

    return mult * (degs + mins / 60)


if __name__ == '__main__':
    with open('sample_gps.txt', 'r') as fh:
        msgs = fh.readlines()

    # Use the process_gps_string func to get a DataFrame with columns 'Lat' and 'Long'
    data = pd.DataFrame([process_gps_string(msg) for msg in msgs], columns=['Lat', 'Long'])

    # Plot the longitudes on the X axis and the Latitudes on the Y axis
    plt.scatter(data['Long'], data['Lat'])
    plt.axis('equal')
    plt.xlim(data['Long'].min(), data['Long'].max())
    plt.ylim(data['Lat'].min(), data['Lat'].max())
    plt.title('GPS Data')
    plt.show()
