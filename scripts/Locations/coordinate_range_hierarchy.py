import numpy as np
from decimal import *

import pandas as pd

getcontext().prec = 15

def generalize(lat,lon):
    # level 0
    lat = Decimal(lat)
    lon = Decimal(lon)
    hierarchy_row = [format_string.format(lat) + ';' + format_string.format(lon)]
    lat_sign = '-' if lat < 0 else '+'
    lon_sign = '-' if lon < 0 else '+'
    for current_range_size in range_sizes:
        lat_range_index = int(abs(lat) / current_range_size)
        lon_range_index = int(abs(lon) / current_range_size)

        lat_max = (lat_range_index + 1) * current_range_size
        lat_min = lat_range_index * current_range_size
        lon_max = (lon_range_index + 1) * current_range_size
        lon_min = lon_range_index * current_range_size

        if lat_max > 90:
            lat_max = 90
        if lon_max > 180:
            lon_max = 180

        lat_center = (lat_max + lat_min) / 2
        lon_center = (lon_max + lon_min) / 2

        lat_center = float(lat_sign + str(lat_center))
        lon_center = float(lon_sign + str(lon_center))

        label = format_string.format(lat_center) + ';' + format_string.format(lon_center)
        hierarchy_row.append(label)
    hierarchy_row.append('*')
    hierarchy_row = ",".join(hierarchy_row) #+ "\n"
    return hierarchy_row

accuracy = 5
format_string = '{:.5f}'
range_sizes = []

def get_parameters():
    acc = ["Accuracy", "int", "5", "Used in the format string"]
    delimiter = ["Delimiter", "string", ";", "The delimiter used in the input data"]
    description = "Script to generate hierarchies of lat-lon coordinates. The script uses a accuracy parameter that determines the number of digits after the decimal point. The delimiter parameter is used to split the lat-lon value. The script will adjust the lat and lon value so that they are within a certain range."
    print(acc)
    print(delimiter)
    print(description)


def create_hierarchy(data:list[str]) -> pd.DataFrame:
    global format_string, range_sizes
    accuracy = int(data[0])
    format_string = "{:." + str(accuracy) + "f}"
    delimiter = data[1]
    start_factor = 0.010986328125
    groupings = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    lat_list = []
    lon_list = []
    for item in data[2:]:
        latitude, longitude = item.split(delimiter)
        lat_list.append(float(latitude))
        lon_list.append(float(longitude))
    X = pd.DataFrame({'latitude': lat_list, 'longitude': lon_list})
    range_sizes = [Decimal(start_factor)]
    for index, group in enumerate(groupings):
        range_sizes.append(group * range_sizes[index])

    generalize_vect = np.vectorize(generalize)
    file_rows = generalize_vect(X['latitude'].values, X['longitude'].values)
    h = pd.DataFrame(columns=range(len(range_sizes)+2))
    for row in file_rows:
        h.loc[len(h)] = row.split(',')
    return h


def print_hierarchy(hierarchy:pd.DataFrame):
    """Print the hierarchy using ',' as the column seperator."""
    for index, row in hierarchy.iterrows():
        print(','.join(row.tolist()))


if __name__ == '__main__':
    data = []
    data_size = int(input())
    for i in range(data_size):
        data.append(input())

    if (data[0] == "getParameters"):
        get_parameters()
    else:
        hierarchy = create_hierarchy(data)
        print_hierarchy(hierarchy)
