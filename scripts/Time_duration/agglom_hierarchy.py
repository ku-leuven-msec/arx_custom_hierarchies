from typing import List
from datetime import datetime
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster._agglomerative import _hc_cut
import ast


def initialize(input_df, columns, multipliers=None):
    data = input_df[columns].copy()
    if multipliers is None:
        multipliers = np.ones(len(columns))

    for column, multiplier in zip(columns, multipliers):
        col = data[column]
        data[f'{column} ml'] = col * multiplier

    return data


def create_hierarchies(data: pd.DataFrame, columns: List[str], cluster_amounts: List[int], multipliers: List[float],
                       generalize_functions, unique:bool=False, init_labels: ndarray = None) -> List[
    pd.DataFrame]:
    cluster_amounts.reverse()

    ml_columns = [f'{column} ml' for column in columns]
    # get data to cluster
    data_to_cluster = data[ml_columns]
    all_columns = columns + ml_columns

    if unique:
        uniques, reverse = np.unique(data_to_cluster.values, axis=0, return_inverse=True)
        data_to_cluster = pd.DataFrame(uniques)

    # cluster in a bottom-up manner
    c_labels = agglomerative_clustering(data_to_cluster, cluster_amounts, init_labels)

    if unique:
        c_labels = [c_label[reverse] for c_label in c_labels]

    hierarchies = []
    for generalize in generalize_functions:
        #print(f'Creating hierarchy: {generalize.__name__}')
        hierarchies.append(generalize(data[all_columns], columns, multipliers, c_labels))

    return hierarchies


def agglomerative_clustering(dataset, amounts, init_labels):
    uniques, reverse = np.unique(dataset.values, return_inverse=True, axis=0)
    model = AgglomerativeClustering(linkage=linkage, metric=metric, n_clusters=None, distance_threshold=0)
    model.fit(uniques)
    labels = [np.zeros(len(dataset),dtype=int)]

    for amount in amounts:
        c_labels = _hc_cut(amount, model.children_, model.n_leaves_).astype(int)[reverse]

        labels.append(c_labels)
    return labels


def output_centers(data, columns, multipliers, c_labels):
    data = data.copy()
    format_string = '{:.5f}'
    ml_columns = [f'{column} ml' for column in columns]

    hierarchy = pd.DataFrame(dtype='<U32')
    hierarchy[0] = data[columns[0]].astype(str) + '::' + data[columns[1]].astype(str)

    data = data[ml_columns]
    data[ml_columns[0]] = ((data[ml_columns[0]] / multipliers[0]) + 0.5).astype(int)
    data[ml_columns[1]] = ((data[ml_columns[1]] / multipliers[1]) + 0.5).astype(int)

    for index, current_labels in enumerate(reversed(c_labels)):
        data['labels'] = current_labels

        centers = data.groupby(by=['labels']).mean()
        centers.columns = ['x_c', 'y_c']

        # string formatter
        centers = centers.applymap(format_string.format)

        merged = pd.merge(data, centers, how='left', left_on='labels', right_index=True)
        hierarchy[index + 1] = merged['x_c'] + '::' + merged['y_c']

    return hierarchy


def output_ranges(data, columns, multipliers, c_labels):
    hierarchy = pd.DataFrame(dtype='<U32')
    hierarchy[0] = data[columns[0]].astype(str) + '::' + data[columns[1]].astype(str)

    # calculate range of all points
    df = pd.DataFrame(dtype=float)
    df['x'] = ((data[f'{columns[0]} ml'] / multipliers[0]) + 0.5).astype(int)
    df['y'] = ((data[f'{columns[1]} ml'] / multipliers[1]) + 0.5).astype(int)

    # calculate ranges for each level
    for index, current_labels in enumerate(reversed(c_labels)):
        df['labels'] = current_labels

        grouped = df.groupby(by=['labels'])
        maxs = grouped[['x', 'y']].max().astype(str)
        mins = grouped[['x', 'y']].min().astype(str)
        ranges_str = '[' + mins['x'] + '-' + maxs['x'] + ']::[' + mins['y'] + '-' + maxs['y'] + ']'
        merged = pd.merge(df, ranges_str.rename('output'), how='left', left_on='labels', right_index=True)

        hierarchy[index + 1] = merged['output']

    return hierarchy



columns = ['Departure Time num', 'Duration']
multipliers = [1.0, 1.0]
linkage = 'ward'
metric = None

cluster_amounts = [7649, 7128, 6240, 5536, 4912, 4564, 3883, 3400, 3286, 2324, 2104, 2050, 2026, 1924, 1461, 1202,
                       1198, 1148, 920, 873, 867, 666, 665, 551, 488, 482, 480, 423, 307, 290, 267, 261, 258, 257, 255,
                       168, 164, 158, 154, 141, 136, 93, 91, 86, 85, 81, 74, 54, 49, 47, 44, 29, 28, 26, 24, 15, 8, 4,
                       2]
data = []

def get_parameters():
    multiplrs = ["Weights", "Array of floats", "[1.0, 1.0]", "Used as weights for the attributes"]
    clstr_amounts = ["Cluster amounts", "Array of integers","[44, 29, 28, 26, 24, 15, 8, 4, 2]", "Different amount of clustersizes" ]
    time_date = ["Date or time", "string", "Date", "Select if the input is a date or a time"]
    delimiter = ["Delimiter", "string", "|", "Delimiter that combines the columns"]
    output_type = ["Output type", "string", "range", "How must each generalization be shown. Options are 'range' or 'center'."]
    description = "Agglomerative Hierarchical Clustering script. It takes data or time values and transforms those into integer values. Those will be used to create a hierarchy in an agglomerative hierarchical way. This means that every value will be its own cluster at the start. At higher levels, clusters will be joined together to create bigger clusters. It has multiple paramters. The first is a weight parameter that will be used to prioritize an attribute. The second is the amount of clusters for each hierarchy level. The next parameter is needed to know which conversion has to be done to the input values. The last parameter is the delimiter that is used to split the location values."
    print(multiplrs)
    print(clstr_amounts)
    print(time_date)
    print(delimiter)
    print(output_type)
    print(description)


def create_hierarchy(data:list[str]) -> pd.DataFrame:
    multipliers = ast.literal_eval(data[0])
    cluster_amounts = ast.literal_eval(data[1])
    filtered = [cluster_amounts[0]]
    for i in cluster_amounts:
        if filtered[-1] * 0.95 > i:
            filtered.append(i)
    cluster_amounts = filtered
    isDate = False
    if data[2] == "Date" or data[2] == "date":
        isDate = True
    delimiter = data[3]
    output_type = None
    if data[4] == 'range':
        output_type = output_ranges
    if data[4] == 'center':
        output_type = output_centers
    if output_type is None:
        print('Non supported output type')
        # return empty hierarchy
        return pd.DataFrame()
    functions = [output_type]
    unique = False

    departTimeNum_list = []
    duration_list = []
    originalValues = []
    for item in data[5:]:
        departTimeNum, duration = item.split(delimiter)
        time_or_date = departTimeNum
        originalValues.append(item)
        if isDate:
            try:
                date = datetime.strptime(departTimeNum, '%d/%m/%Y')
                time_or_date = date.timetuple().tm_yday
            except Exception as e: print(e)


        else:
            time = departTimeNum.split(':')
            time_or_date = int(time[0]) * 60 + int(time[1])
        departTimeNum_list.append(int(time_or_date))
        duration_list.append(int(duration))
    X = pd.DataFrame({'Departure Time num': departTimeNum_list, 'Duration': duration_list})
    input_df = initialize(X, columns, multipliers)
    init_labels = None

    # The code for the paper originaly printed multiple hierarchies having various output formats to files
    hierarchies = create_hierarchies(input_df, columns, cluster_amounts, multipliers, functions, unique=unique, init_labels=init_labels)
    h = hierarchies[0]
    # add original value level
    h.insert(0, 'original', originalValues)
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