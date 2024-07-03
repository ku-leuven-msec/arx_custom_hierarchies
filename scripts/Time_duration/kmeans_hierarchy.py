import os

os.environ['OMP_NUM_THREADS'] = '1'
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree
from datetime import datetime
from multiprocessing import Pool
from typing import List, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import ast
from k_means_constrained import KMeansConstrained
from numpy import ndarray
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


# flight_fare.clustering.utils import initialize

def initialize(input_df, columns, multipliers=None):
    data = input_df[columns].copy()
    if multipliers is None:
        multipliers = np.ones(len(columns))

    for column, multiplier in zip(columns, multipliers):
        col = data[column]
        data[f'{column} ml'] = col * multiplier

    return data


def summed_distances_fast(mat1, mat2):
    unique1, inverse1, counts1 = np.unique(mat1, return_inverse=True, return_counts=True, axis=0)
    unique2, inverse2, counts2 = np.unique(mat2, return_inverse=True, return_counts=True, axis=0)

    distances = pairwise_distances(unique1, unique2, metric='euclidean')
    summed_distances = np.sum(np.multiply(distances, counts2), axis=1)
    return summed_distances[inverse1]


def nnit(mat, clsize=10, method='random'):
    clsize = np.ceil(np.arange(1, mat.shape[0] + 1) / (mat.shape[0] / clsize)).astype(int)
    clsize = np.bincount(clsize)
    lab = np.full(mat.shape[0], np.nan, dtype=float)

    # init sum of distances
    if method == 'maxd' or method == 'mind':
        distance_sums = summed_distances_fast(mat, mat)

    cpt = 0
    while np.isnan(lab).sum() > 0:
        lab_ii = np.where(np.isnan(lab))[0]
        if method == 'random':
            ii = np.random.choice(len(lab_ii), 1)[0]
        elif method == 'maxd':
            ii = np.argmax(distance_sums)
        elif method == 'mind':
            ii = np.argmin(distance_sums)
        else:
            raise ValueError('unknown method')

        lab_m = np.full(lab_ii.shape, np.nan, dtype=float)

        # calculate clsize[cpt] nearest neighbors of mat[ii] to mat[lab_ii] using kdtree
        tree = KDTree(mat[lab_ii])
        indishes = tree.query(mat[ii].reshape(1, 2), k=clsize[cpt + 1], return_distance=False)[0]

        lab_m[indishes] = cpt
        lab[lab_ii] = lab_m

        if method == 'maxd' or method == 'mind':
            # remove indices rows
            distance_sums = np.delete(distance_sums, indishes)
            # remove distance of leftover to all indices points
            if len(distance_sums) != 0:
                to_remove = summed_distances_fast(mat[np.delete(lab_ii, indishes)], mat[lab_ii[indishes]])
                distance_sums -= to_remove
        cpt += 1
    if np.isnan(lab).sum() > 0:
        lab[np.where(np.isnan(lab))[0]] = cpt
    return lab.astype(int)


def create_hierarchies(data: pd.DataFrame, columns: List[str], cluster_amounts: List[int], multipliers: List[float],
                       generalize_functions, unique: bool = False, init_labels: ndarray = None) -> List[
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
        # print(f'Creating hierarchy: {generalize.__name__}')
        hierarchies.append(generalize(data[all_columns], columns, multipliers, c_labels))

    return hierarchies


def apply_kmeans(dataset, n_clusters):
    mbk = MiniBatchKMeans(n_clusters=n_clusters, n_init=5, reassignment_ratio=0)
    labels = mbk.fit_predict(dataset)

    # removes skipped labels when empty cluster are created
    return np.unique(labels, return_inverse=True)[1]


def apply_xmeans(dataset, n_clusters):
    amount_initial_centers = 2
    initial_centers = kmeans_plusplus_initializer(dataset, amount_initial_centers).initialize()
    xmeans_instance = xmeans(dataset, initial_centers, n_clusters)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    labels = np.zeros(len(dataset), dtype=int)

    for cluster_label, cluster in enumerate(clusters):
        for row in cluster:
            labels[row] = cluster_label
    return labels


def get_min_cluster_size(dataset, n_clusters):
    # calculates size for equal distribution, it can occure that this causes a large set of identical points to be split
    # this is not allowed, so we recalculate the size taking this into account
    counts = np.sort(np.unique(dataset, axis=0, return_counts=True)[1])[::-1]
    data_size = len(dataset)
    min_size = int(data_size / n_clusters)
    remove_candidate = 0

    while min_size < counts[remove_candidate]:
        data_size -= counts[remove_candidate]
        n_clusters -= 1
        min_size = int(data_size / n_clusters)
        remove_candidate += 1
    return min_size


def apply_kmeans_constrained(dataset, n_clusters):
    size_min = get_min_cluster_size(dataset, n_clusters)
    # size_min = int(len(dataset) / n_clusters)
    mbk = KMeansConstrained(n_clusters=n_clusters, n_init=5, size_min=size_min)
    labels = mbk.fit_predict(dataset)

    # this algorithm can assign multiple equal points to different labels
    row_to_label_counts = defaultdict(lambda: defaultdict(int))
    for a, b in zip(dataset.tolist(), labels):
        row_to_label_counts[str(a)][b] += 1

    # smart assign grote groep van gelijke punten aan kleinste cluster
    row_to_label = {}
    current_counts = defaultdict(int)
    leftover_groups = {}
    for row, clusters in row_to_label_counts.items():
        if len(clusters.keys()) == 1:
            key, value = list(clusters.items())[0]
            current_counts[key] += value
            row_to_label[row] = key
        else:
            clusters['total'] = sum(clusters.values())
            leftover_groups[row] = clusters

    for row, clusters in sorted(leftover_groups.items(), key=lambda x: x[1]['total'], reverse=True):
        possible_labels = list(clusters.keys())
        possible_labels.remove('total')
        new_label = min(possible_labels, key=current_counts.__getitem__)

        row_to_label[row] = new_label
        current_counts[new_label] += clusters['total']

    df = pd.DataFrame(dataset)
    labels = df.apply(lambda row: row_to_label[str(row.tolist())], axis=1).values

    return labels


def apply_nn_constrained(dataset, n_clusters):
    labels = nnit(dataset, n_clusters, 'maxd')

    # this algorithm can assign multiple equal points to different labels
    row_to_label_counts = defaultdict(lambda: defaultdict(int))
    for a, b in zip(dataset.tolist(), labels):
        row_to_label_counts[str(a)][b] += 1

    # smart assign grote groep van gelijke punten aan kleinste cluster
    row_to_label = {}
    current_counts = defaultdict(int)
    leftover_groups = {}
    for row, clusters in row_to_label_counts.items():
        if len(clusters.keys()) == 1:
            key, value = list(clusters.items())[0]
            current_counts[key] += value
            row_to_label[row] = key
        else:
            clusters['total'] = sum(clusters.values())
            leftover_groups[row] = clusters

    for row, clusters in sorted(leftover_groups.items(), key=lambda x: x[1]['total'], reverse=True):
        possible_labels = list(clusters.keys())
        possible_labels.remove('total')
        new_label = min(possible_labels, key=current_counts.__getitem__)

        row_to_label[row] = new_label
        current_counts[new_label] += clusters['total']

    df = pd.DataFrame(dataset)
    labels = df.apply(lambda row: row_to_label[str(row.tolist())], axis=1).values

    return labels


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

        new_cluster_sizes[label] = c_sizes[label] / sub_clusters[label]

        if len(cant_split_more) == len(c_sizes):
            print('FUCK it happened, cannot split', needed_clusters)
            break

        if sub_clusters[label] == uniques[label]:
            cant_split_more.add(label)

    return sub_clusters


def recluster(input_date):
    amount, data = input_date
    if amount == 0:
        print('a subcluster amount of 0 occurred')
    elif amount == 1:
        return np.zeros(len(data), dtype=int)
    else:
        return cluster_algorithm(data, amount)


def recluster_parallel(dataset_np, new_cluster_amounts, prev_labels):
    c_labels = np.full(len(dataset_np), -1)

    dataset_df = pd.DataFrame(dataset_np)
    dataset_df['prev_labels'] = prev_labels
    grouped = dataset_df.groupby(by='prev_labels')[list(dataset_df.columns[:-1])]

    jobs = [(amount, grouped.get_group(label[0]).values) for label, amount in np.ndenumerate(new_cluster_amounts)]
    results = []

    if parallel:
        with Pool(processes=cores) as pool:
            max_ = np.sum(new_cluster_amounts)
            with tqdm(total=max_) as pbar:
                for result in pool.imap(recluster, jobs):
                    results.append(result)
                    pbar.update(len(np.unique(result)))
    else:
        for job in jobs:
            results.append(recluster(job))

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

        c_labels = recluster_parallel(dataset_np, new_cluster_amounts, labels[-1])
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


# does not print a hierarchy but a level,label to distribution mapping. printing in hierarchy was already 1.6GB for the small range set
def output_distr(data, columns, multipliers, c_labels):
    data = data.copy()
    ml_columns = [f'{column} ml' for column in columns]

    distr = pd.DataFrame(dtype='<U32')

    data = data[ml_columns]
    data[ml_columns[0]] = ((data[ml_columns[0]] / multipliers[0]) + 0.5).astype(int)
    data[ml_columns[1]] = ((data[ml_columns[1]] / multipliers[1]) + 0.5).astype(int)

    for index, current_labels in enumerate(reversed(c_labels)):
        data['labels'] = current_labels

        grouped = data.groupby(by=['labels'])
        col_distr = []
        for col in ml_columns:
            col_distr.append(grouped[col].value_counts())

        level_distr = []
        for label in range(current_labels.max() + 1):
            level_distr.append([distr[label].to_dict() for distr in col_distr])
        distr[index] = pd.Series(level_distr)

    return distr


def output_labels(data, columns, multipliers, c_labels):
    hierarchy = pd.DataFrame(dtype='<U32')
    hierarchy[0] = data[columns[0]].astype(str) + '::' + data[columns[1]].astype(str)

    for index, current_labels in enumerate(reversed(c_labels)):
        hierarchy[index + 1] = current_labels

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


def algorithmChoice(choice):
    switch = {
        "kmeans_constrained": apply_kmeans_constrained,
        "kmeans": apply_kmeans,
        "xmeans": apply_xmeans,
        "nn_constrained": apply_nn_constrained
    }
    return switch.get(choice, "Invalid input")


# PAPER USES kmeans and nn, others also work but results where less interesting
# cluster_algorithm = apply_kmeans_constrained
# cluster_algorithm = apply_kmeans
# cluster_algorithm = apply_xmeans
cluster_algorithm = apply_nn_constrained
cores = 8
parallel = False
multipliers = [1.0, 1.0]
cluster_amounts = [7649, 7128, 6240, 5536, 4912, 4564, 3883, 3400, 3286, 2324, 2104, 2050, 2026, 1924, 1461, 1202,
                   1198,
                   1148, 920, 873, 867, 666, 665, 551, 488, 482, 480, 423, 307, 290, 267, 261, 258, 257, 255, 168,
                   164,
                   158, 154, 141, 136, 93, 91, 86, 85, 81, 74, 54, 49, 47, 44, 29, 28, 26, 24, 15, 8, 4, 2]

def get_parameters():
    algo = ["Cluster algorithm", "String", "apply_nn_constrained", "Algorithm used for clustering. Options are 'kmeans_constrained', 'kmean', 'xmean' and 'nn_constrained'."]
    crs = ["Amount of cores", "int", "2", "Amount of cores the script may use"]
    parallelInpt = ["Parallel", "bool", "False", "Indicates if program can run functions in parallel"]
    multiplrs = ["Multipliers", "Array of floats", "[1.0, 1.0]", "Used as weight for attributes"]
    cluster_amounts_inputs = ["Cluster amounts", "Array of integers", "[44, 29, 28, 26, 24, 15, 8, 4, 2]",
                                "Different amount of clustersizes"]
    time_date = ["Date or time", "string", "Date", "Select if the input is a date or a time"]
    delimiter = ["Delimiter", "string", "|", "Delimiter that combines the columns"]
    output_type = ["Output type", "string", "range", "How must each generalization be shown. Options are 'range' or 'center'."]
    description = "Script that creates hierarchies for dates or times and duratios. There are 4 different clustering algorithms to choose from: apply_nn_constrained, apply_kmeans_constrained, apply_kmeans, apply_xmeans. Choose between the for by editing the clustering algorithm parameter. The script can be run in parallel by setting the parallel parameter to True. The multipliers parameter is used as weights for the attributes. The cluster amounts parameter is used to determine the amount of clusters that will be created. The time_date parameter is used to determine if the input data is a date or a time. The delimiter parameter is used to determine the delimiter that combines the columns. The script will return the hierarchy values per level per input value."
    print(algo)
    print(crs)
    print(parallelInpt)
    print(multiplrs)
    print(cluster_amounts_inputs)
    print(time_date)
    print(delimiter)
    print(output_type)
    print(description)


def create_hierarchy(data:list[str]) -> pd.DataFrame:
    global cluster_algorithm, cores, parallel, multipliers
    cluster_algorithm = algorithmChoice(data[0])
    if cluster_algorithm == "Invalid input":
        print('Invalid algorithm')
        return pd.DataFrame()
    cores = int(data[1])
    if ("True" == data[2]):
        parallel = True
    multipliers = ast.literal_eval(data[3])
    cluster_amounts = ast.literal_eval(data[4])
    isDate = False
    if data[5] == "Date" or data[5] == "date":
        isDate = True
    delimiter = data[6]
    columns = ['Departure Time num', 'Duration']

    if data[7] == 'range':
        output_type = output_ranges
    if data[7] == 'center':
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

    unique = False

    departTimeNum_list = []
    duration_list = []
    originalValues = []
    for item in data[8:]:
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
    hierarchies = create_hierarchies(input_df, columns, cluster_amounts, multipliers, functions, unique=unique,
                                        init_labels=init_labels)

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