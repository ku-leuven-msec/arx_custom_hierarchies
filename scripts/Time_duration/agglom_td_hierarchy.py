import multiprocessing
import os
from multiprocessing import Pool
from typing import List, Any
from datetime import datetime
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.cluster import AgglomerativeClustering
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
                       generalize_functions, delimiter:str, unique:bool=False, init_labels: ndarray = None) -> List[
    pd.DataFrame]:
    cluster_amounts.reverse()

    ml_columns = [f'{column} ml' for column in columns]
    # get data to cluster
    data_to_cluster = data[ml_columns]
    all_columns = columns + ml_columns

    if unique:
        uniques, reverse = np.unique(data_to_cluster.values, axis=0, return_inverse=True)
        data_to_cluster = pd.DataFrame(uniques)

    # cluster in a top-down manner
    c_labels = top_down_clustering(data_to_cluster, cluster_amounts, init_labels)

    if unique:
        c_labels = [c_label[reverse] for c_label in c_labels]

    hierarchies = []
    for generalize in generalize_functions:
        #print(f'Creating hierarchy: {generalize.__name__}')
        hierarchies.append(generalize(data[all_columns], columns, multipliers, c_labels, delimiter))

    return hierarchies


def agglom(dataset, n_clusters):
    uniques, reverse = np.unique(dataset, return_inverse=True, axis=0)

    linkage = 'ward'
    model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
    labels = model.fit_predict(uniques)[reverse]
    # removes skipped labels when empty cluster are created
    return np.unique(labels, return_inverse=True)[1]


def get_new_cluster_amounts(np_dataset, prev_labels, needed_clusters):
    """Calculates for each previous cluster how many sub-clusters are needed,
     based on their current size in comparison to the total size"""

    # calculate cluster sizes
    c_sizes = np.bincount(prev_labels)
    total_records = len(np_dataset)

    # calculate num_unique for each cluster
    uniques = np.zeros(len(c_sizes), dtype=int)
    for label in range(len(c_sizes)):
        uniques[label] = len(np.unique(np_dataset[prev_labels == label], axis=0))

    # a prev cluster can't be devided into more clusters than unique values it has
    merged = np.vstack((c_sizes, uniques)).T
    cant_split_more = set()
    sub_clusters = np.apply_along_axis(lambda row: min(max(int((row[0] / total_records) * needed_clusters), 1), row[1]),
                                       1, merged)
    cant_split_more.update(np.nonzero(np.equal(uniques, sub_clusters))[0])

    # more groups will be needed, split current largest subgroups more
    new_cluster_sizes = np.divide(c_sizes, sub_clusters)

    all_labels = np.arange(len(c_sizes))
    while np.sum(sub_clusters) < needed_clusters:
        allowed_labels = np.where(~np.isin(all_labels, list(cant_split_more)))[0]
        label = allowed_labels[new_cluster_sizes[allowed_labels].argmax()]
        sub_clusters[label] += 1

        if sub_clusters[label] == uniques[label]:
            cant_split_more.add(label)
        new_cluster_sizes[label] = c_sizes[label] / sub_clusters[label]

        if len(cant_split_more) == len(c_sizes):
            print('Cannot split', needed_clusters)
            break

    return sub_clusters


def recluster(input_date):
    amount, data = input_date
    if amount == 0:
        print('A subcluster amount of 0 occurred')
    elif amount == 1:
        return np.zeros(len(data),dtype=int)
    else:
        return agglom(data, amount)


def recluster_parallel(dataset_np, new_cluster_amounts, prev_labels):
    c_labels = np.full(len(dataset_np), -1)

    dataset_df = pd.DataFrame(dataset_np)
    dataset_df['prev_labels'] = prev_labels
    grouped = dataset_df.groupby(by='prev_labels')[list(dataset_df.columns[:-1])]

    jobs = [(amount,grouped.get_group(label[0]).values) for label,amount in np.ndenumerate(new_cluster_amounts)]
    results = []
    multiprocessing.freeze_support()
    with Pool(processes=cores) as pool:
        max_ = np.sum(new_cluster_amounts)
        #with tqdm(total=max_) as pbar:
        for result in pool.imap(recluster, jobs):
            results.append(result)
                #pbar.update(len(np.unique(result)))

    max_label = 0
    for label, new_amount in np.ndenumerate(new_cluster_amounts):
        label = label[0]
        data_filter = prev_labels == label
        current_labels = results[label]

        if (a := len(set(current_labels))) != new_cluster_amounts[label]:
            print('Cluster amount missmatch. Got: ', a, 'expected: ', new_cluster_amounts[label])

        current_max_label = np.max(current_labels)
        current_labels += max_label
        max_label += current_max_label + 1

        c_labels[data_filter] = current_labels
    return c_labels


def top_down_clustering(dataset, cluster_amounts, init_labels: List[int] = None) -> list[
    ndarray | Any]:
    if init_labels is not None and (init_label_amount := len(set(init_labels))) >= np.min(cluster_amounts):
        raise ValueError(
            f'There are more unique initial labels than the smallest cluster. Expected <{np.min(cluster_amounts)} got {init_label_amount}')

    labels = []
    dataset_np = dataset.values

    if init_labels is not None:
        # set the initial labels as the first clustering level
        labels.append(np.array(init_labels))
    else:
        # first level just assign every point label 0 (suppression level)
        labels.append(np.zeros(len(dataset), dtype=int))

    # all other clusters, apply clustering on each previous level cluster
    for amount in cluster_amounts:
        # calculate for each cluster, how many sub-clusters need to be generated
        new_cluster_amounts = get_new_cluster_amounts(dataset_np, labels[-1], amount)

        c_labels = recluster_parallel(dataset_np,new_cluster_amounts,labels[-1])
        labels.append(c_labels)

    return labels


def output_centers(data, columns, multipliers, c_labels, delimiter):
    data = data.copy()
    format_string = '{:.5f}'
    ml_columns = [f'{column} ml' for column in columns]

    hierarchy = pd.DataFrame(dtype='<U32')
    hierarchy[0] = data[columns[0]].astype(str) + delimiter + data[columns[1]].astype(str)

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
        hierarchy[index + 1] = merged['x_c'] + delimiter + merged['y_c']

    return hierarchy


def output_ranges(data, columns, multipliers, c_labels,delimiter):
    hierarchy = pd.DataFrame(dtype='<U32')
    hierarchy[0] = data[columns[0]].astype(str) + delimiter + data[columns[1]].astype(str)

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
        ranges_str = '[' + mins['x'] + '-' + maxs['x'] + ']'+delimiter+'[' + mins['y'] + '-' + maxs['y'] + ']'
        merged = pd.merge(df, ranges_str.rename('output'), how='left', left_on='labels', right_index=True)

        hierarchy[index + 1] = merged['output']

    return hierarchy

def get_parameters():
    multiplrs = ["Multipliers", "Array of floats", "[1.0, 1.0]", "Used as a weights for the columns"]
    clstr_amounts = ["Cluster amounts", "Array of integers","[44, 29, 28, 26, 24, 15, 8, 4, 2]", "Different amount of clustersizes" ]
    crs = ["Amount of cores", "int", "2", "Amount of cores the script may use"]
    threads = ["Amount of threads", "int", "4", "Amount of threads the script can use"]
    time_date = ["Date or time", "string", "Date", "Select if the input is a date or a time"]
    delimiter = ["Delimiter", "string", "|", "Delimiter that combines the columns"]
    output_type = ["Output type", "string", "range", "How must each generalization be shown. Options are 'range' or 'center'."]
    description = "Agglomerative Hierarchical Clustering script (top-down). In this script, items that have similar values will be clustered together. The value of the cluster will be a combination of the values of the items that are assigned to that cluster. A tree structure is used when clustering the values. The agglomerative method means that every value starts as its own cluster. When building the tree structure, the clusters that are close to each other will be combined. This creates a new level in the tree. This can go on until there is only one cluster."

    print(multiplrs)
    print(clstr_amounts)
    print(crs)
    print(threads)
    print(time_date)
    print(delimiter)
    print(output_type)
    print(description)

cores = 4
def create_hierarchy(data:list[str]) -> pd.DataFrame:
    global cores
    columns = ['Departure Time num', 'Duration']

    unique = False
    multipliers = ast.literal_eval(data[0])
    cluster_amounts = ast.literal_eval(data[1])
    cores = int(data[2])
    os.environ['OMP_NUM_THREADS'] = data[3]
    isDate = False
    if data[4] == "Date" or data[4] == "date":
        isDate = True
    delimiter = data[5]

    if data[6] == 'range':
        output_type = output_ranges
    if data[6] == 'center':
        output_type = output_centers
    if output_type is None:
        print('Non supported output type')
        # return empty hierarchy
        return pd.DataFrame()

    filtered = [cluster_amounts[0]]
    for i in cluster_amounts:
        if filtered[-1] * 0.95 > i:
            filtered.append(i)
    cluster_amounts = filtered

    functions = [output_type]

    departTimeNum_list = []
    duration_list = []
    originalValues = []
    for item in data[7:]:
        departTimeNum, duration = item.split(delimiter)
        time_or_date = departTimeNum
        originalValues.append(item)
        if isDate:
            date = datetime.strptime(departTimeNum, '%d/%m/%Y')
            time_or_date = date.timetuple().tm_yday
        else:
            time = departTimeNum.split(':')
            time_or_date = int(time[0]) * 60 + int(time[1])
        departTimeNum_list.append(int(time_or_date))
        duration_list.append(int(duration))
    X = pd.DataFrame({'Departure Time num': departTimeNum_list, 'Duration': duration_list})

    input_df = initialize(X, columns, multipliers)
    # init_labels = init_labels_season_ltsa(input_df)
    init_labels = None

    # The code for the paper originaly printed multiple hierarchies having various output formats to files
    hierarchies = create_hierarchies(input_df, columns, cluster_amounts, multipliers, functions, delimiter, unique=unique, init_labels=init_labels)

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