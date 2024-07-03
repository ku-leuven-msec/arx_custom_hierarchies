import numpy as np
import pandas as pd

def generalize(lat,lon):
    lat_original = format_string.format(lat)
    lon_original = format_string.format(lon)

    lat_general = lat_original.replace('-', '')
    lon_general = lon_original.replace('-', '')

    lat_general = lat_general.zfill(max_clean_length)
    lon_general = lon_general.zfill(max_clean_length)

    lat_sign = '-' if lat < 0 else '+'
    lon_sign = '-' if lon < 0 else '+'

    lat_general = lat_sign + lat_general
    lon_general = lon_sign + lon_general

    coordinates_original = lat_original + ';' + lon_original
    new_row = [coordinates_original]

    lat_general = list(lat_general)
    lon_general = list(lon_general)
    for i in reversed(range(len(lat_general))):
        if lat_general[i] == '.':
            continue

        lat_general[i] = '*'
        current_lat_general = "".join(lat_general)

        lon_general[i] = '*'
        current_lon_general = "".join(lon_general)

        new_row.append(current_lat_general + ';' + current_lon_general)
    return ','.join(new_row)

accuracy = 5
max_clean_length = 9
format_string = '{:.5f}'

import pandas as pd


def get_parameters():
    acc = ["Accuracy", "int", "5", "The number of digits after the decimal point"]
    delimiter = ["Delimiter", "string", ";", "The delimiter used in the input data"]
    description = "Script that creates a hierarchy for a lat-lon location by removing the least significant number each hierarchy level. The script uses a accuracy parameter that determines the number of digits after the decimal point. The delimiter parameter is used to split the lat-lon value. The script will output the hierarchy levels in the following format: lat;lon, lat;lon, lat;lon. The first value is the original lat;lon location, the second value is the location with the least significant digit removed, the third value is the location with the two least significant digits removed, etc. "
    print(acc)
    print(delimiter)
    print(description)


def create_hierarchy(data:list[str]) -> pd.DataFrame:
    global max_clean_length, format_string
    accuracy = int(data[0])
    delimiter = data[1]
    max_clean_length = 4 + accuracy
    format_string = "{:." + str(accuracy) + "f}"
    lat_list = []
    lon_list = []

    for item in data[2:]:
        latitude, longitude = item.split(delimiter)
        lat_list.append(float(latitude))
        lon_list.append(float(longitude))
    X = pd.DataFrame({'latitude': lat_list, 'longitude': lon_list})
    generalize_vect = np.vectorize(generalize)
    output_rows = generalize_vect(X['latitude'].values, X['longitude'].values)

    h = pd.DataFrame(columns=range(max_clean_length))
    for row in output_rows:
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
