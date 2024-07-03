import numpy as np
from decimal import *

import pandas as pd

getcontext().prec = 15



levels = 10
divider = 2
accuracy = 5


def get_parameters():
    lvls = ["Levels", "int", "10", "Amount of hierarchy levels"]
    div = ["Divider", "int", "2", "Divider for each level"]
    acc = ["Accuracy", "int", "5", "Determines the number of digits after the decimal point"]
    description = "Script that creates hierarchies using the lat and lon coordinates. The script uses the accuracy parameter to determine the number of digits after the decimal point. The levels parameter determines the amount of hierarchy levels. The divider parameter is used to divide the range in each level. The script will output the hierarchy levels in the following format: lat;lon, lat;lon, lat;lon."
    print(lvls)
    print(div)
    print(acc)
    print(description)


def create_hierarchy(data:list[str]) -> pd.DataFrame:
    levels = int(data[0])
    divider = int(data[1])
    accuracy = int(data[2])
    precision = Decimal(1) / Decimal((10 ** accuracy))
    format_string = "{:." + str(accuracy) + "f}"

    lat_list = []
    lon_list = []

    for item in data[3:]:
        latitude, longitude = item.split(';')
        lat_list.append(float(latitude))
        lon_list.append(float(longitude))
    X = pd.DataFrame({'latitude': lat_list, 'longitude': lon_list})

    range_sizes = []
    lat_ranges = {}
    lon_ranges = {}
    # get start range size (max dimension of country)
    max_lat = Decimal(format_string.format(X['latitude'].max())) + precision
    min_lat = Decimal(format_string.format(X['latitude'].min()))
    lat_range_size = max_lat - min_lat
    max_lon = Decimal(format_string.format(X['longitude'].max())) + precision
    min_lon = Decimal(format_string.format(X['longitude'].min()))
    lon_range_size = max_lon - min_lon
    start_range = max(lon_range_size, lat_range_size)

    # shift
    if start_range == lon_range_size:
        min_lat = min_lat + lat_range_size / 2 - start_range / 2
    else:
        min_lon = min_lon + lon_range_size / 2 - start_range / 2

    range_sizes.append(start_range)

    # get first range min and max (starting from current min lat and long)
    lon_range = (min_lon, min_lon + start_range)
    lon_centers = {0: [(lon_range[0] + lon_range[1]) / 2]}
    lon_ranges[0] = [lon_range]
    lat_range = (min_lat, min_lat + start_range)
    lat_centers = {0: [(lat_range[0] + lat_range[1]) / 2]}
    lat_ranges[0] = [lat_range]

    # divide previous ranges by 2 each time and safe in ranges dictionary
    for level in range(1, levels):
        new_lat_ranges = []
        new_lat_centers = []
        new_range_size = range_sizes[level - 1] / divider
        range_sizes.append(new_range_size)
        for previous_lat_range in lat_ranges[level - 1]:
            min_range = previous_lat_range[0]
            for part in range(divider):
                max_range = min_range + new_range_size
                if max_range > previous_lat_range[1]:
                    max_range = previous_lat_range[1]
                new_range = (min_range, max_range)
                new_lat_centers.append((min_range + max_range) / 2)
                new_lat_ranges.append(new_range)
                min_range = max_range
        lat_ranges[level] = new_lat_ranges
        lat_centers[level] = new_lat_centers

        new_lon_ranges = []
        new_lon_centers = []
        for previous_lon_range in lon_ranges[level - 1]:
            min_range = previous_lon_range[0]
            for part in range(divider):
                max_range = min_range + new_range_size
                if max_range > previous_lon_range[1]:
                    max_range = previous_lon_range[1]
                new_range = (min_range, max_range)
                new_lon_ranges.append(new_range)
                new_lon_centers.append((min_range + max_range) / 2)
                min_range = max_range
        lon_ranges[level] = new_lon_ranges
        lon_centers[level] = new_lon_centers

    # find for every location the corresponding range in each level and save the center as label
    # levels are reverse build compared to hierarchy

    def generalize(lat,lon):
        lat = format_string.format(lat)
        lon = format_string.format(lon)
        hierarchy_row = [lat + ';' + lon]
        lat = Decimal(lat)
        lon = Decimal(lon)
        for level in reversed(range(levels)):
            # get latitude range
            lat_range_index = int((lat - min_lat) / range_sizes[level])
            lat_center = lat_centers[level][lat_range_index]
            # get longitude range
            lon_range_index = int((lon - min_lon) / range_sizes[level])
            lon_center = lon_centers[level][lon_range_index]
            hierarchy_row.append(format_string.format(lat_center) + ';' + format_string.format(lon_center))
        hierarchy_row = ",".join(hierarchy_row)# + "\n"
        return hierarchy_row

    
    generalize_vect = np.vectorize(generalize)
    file_rows = generalize_vect(X['latitude'].values, X['longitude'].values)
    h = pd.DataFrame(columns=range(levels+1))
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
        