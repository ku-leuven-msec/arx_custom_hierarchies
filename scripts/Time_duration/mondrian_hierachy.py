import ast
from typing import List
from datetime import datetime
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from numpy import ndarray

import Mondrian_clustering as mc

def initialize(input_df, columns, multipliers=None):
    data = input_df[columns].copy()
    if multipliers is None:
        multipliers = np.ones(len(columns))

    for column, multiplier in zip(columns, multipliers):
        col = data[column]
        data[f'{column} ml'] = col * multiplier

    return data

def cell_ranges(data, labels: List[str], model: mc.MondrianClustering):
    hierarchy = pd.DataFrame(dtype='<U32')

    hierarchy[0] = data[labels[0]].astype(str) + '::' + data[labels[1]].astype(str)

    def range_to_string(current_range):
        x0, y0, x1, y1 = current_range
        return np.array([f'[{x0}-{x1}]::[{y0}-{y1}]'], dtype='<U32')

    for level in reversed(range(len(model.ranges))):
        c_labels = model.labels[level]
        ranges = model.ranges[level]
        ranges_str = np.apply_along_axis(range_to_string, 1, ranges)
        hierarchy[len(model.ranges) - level] = ranges_str[c_labels]
    return hierarchy


def output_ranges(data, columns: List[str], model: mc.MondrianClustering):
    hierarchy = pd.DataFrame(dtype='<U32')
    hierarchy[0] = data[columns[0]].astype(str) + '::' + data[columns[1]].astype(str)

    # calculate range of all points
    ml_columns = [f'{column} ml' for column in columns]
    df = pd.DataFrame(dtype=int)
    df['x'], df['y'] = data[ml_columns[0]].astype(int), data[ml_columns[1]].astype(int)

    # calculate ranges for each level
    for index, current_labels in enumerate(reversed(model.labels)):
        df['labels'] = current_labels

        grouped = df.groupby(by=['labels'])
        maxs = grouped[['x', 'y']].max().astype(str)
        mins = grouped[['x', 'y']].min().astype(str)
        ranges_str = '[' + mins['x'] + '-' + maxs['x'] + ']::[' + mins['y'] + '-' + maxs['y'] + ']'
        merged = pd.merge(df, ranges_str.rename('output'), how='left', left_on='labels', right_index=True)

        hierarchy[index + 1] = merged['output']

    return hierarchy


def create_hierarchies(data: pd.DataFrame, columns: List[str], cluster_amounts: List[int],
                       weights: List[float],
                       cut_threshold: float, relax_steps: int, dim_strategy: mc.DimSelectionStrategy, max_best: str,
                       generalize_functions, unique: bool = False, init_labels: ndarray = None) -> List[pd.DataFrame]:
    # get data to cluster
    data_to_cluster = data[[f'{column} ml' for column in columns]]

    if unique:
        uniques, reverse = np.unique(data_to_cluster.values, axis=0, return_inverse=True)
        data_to_cluster = pd.DataFrame(uniques)

    # cluster using mondrian
    model = mc.MondrianClustering(cluster_amounts, cut_threshold=cut_threshold, auto_relax_steps=relax_steps,
                                  dim_strategy=dim_strategy, max_best=max_best)
    #model.fit_predict(data_to_cluster, target, weights=weights, init_labels=init_labels, progress=False)
    model.fit_predict(data_to_cluster, weights=weights, init_labels=init_labels, progress=False)

    if unique:
        new_labels = np.full(shape=(model.max_levels, len(data)), fill_value=-1, dtype=int)
        for level in range(len(new_labels)):
            new_labels[level] = model.labels[level][reverse]
        model.labels = new_labels

    hierarchies = []
    for generalize in generalize_functions:
        # print(f'Creating hierarchy: {generalize.__name__}')
        hierarchies.append(generalize(data, columns, model))

    return hierarchies

def get_strat(strat):
    switch = {
        "ENTROPY": mc.DimSelectionStrategy.ENTROPY,
        "RELATIVE_RANGE": mc.DimSelectionStrategy.RELATIVE_RANGE,
        "UNIFORMITY": mc.DimSelectionStrategy.UNIFORMITY,
        "MSE": mc.DimSelectionStrategy.MSE
    }
    return switch.get(strat, "Invalid input")


def get_parameters():
    cluster_amounts_input = ["Cluster amounts", "int[]", "[44, 29, 28, 26, 24, 15, 8, 4, 2]", "The amount of clusters per level"]
    cut_threshold_input = ["Cut threshold", "int", "0", "Adjusts the the split"]
    relax_steps_input = ["Relax steps", "int", "0", "Sets the amount of steps until no more splits can be made"]
    dim_strategy_input = ["Dim strategy", "String", "UNIFORMITY", "Sets the strategy that will be used. Options: ENTROPY, RELATIVE_RANGE, UNIFORMITY, MSE"]
    max_best_input = ["Max best", "String", "max", "Sets the strategy implementation. Options: max, best"]
    weights_input = ["Weights", "Array of floats", "[1.0, 1.0]", "Used as weight for attributes"]
    time_date = ["Date or time", "string", "Date", "Select if the input is a date or a time"]
    delimiter = ["Delimiter", "string", "|", "Delimiter that combines the columns"]
    output_type = ["Output type", "string", "range", "How must each generalization be shown. Options are 'range' or 'cell_range'."]
    description = "Script that implements a clustering technique based on the mondrian k-anonimity algorithm to create hierarchies. It splits the dataset in different parts. Where the split happens depends on the chosen strategy."
    print(cluster_amounts_input)
    print(cut_threshold_input)
    print(relax_steps_input)
    print(dim_strategy_input)
    print(max_best_input)
    print(weights_input)
    print(time_date)
    print(delimiter)
    print(output_type)
    print(description)


def create_hierarchy(data:list[str]) -> pd.DataFrame:
    columns = ['Departure Time num', 'Duration']
    cluster_amounts = ast.literal_eval(data[0])
    cut_threshold = int(data[1])
    relax_steps = int(data[2])
    strat = get_strat(data[3])
    best_max = "best"
    if data[4] == "max" or data[4] == "Max":
        best_max = "max"
    weights = ast.literal_eval(data[5])
    isDate = False
    if data[6] == "Date" or data[6] == "date":
        isDate = True
    delimiter = data[7]

    if data[8] == 'range':
        output_type = output_ranges
    if data[8] == 'cell_range':
        output_type = cell_ranges
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
    originalValues = []
    departTimeNum_list = []
    duration_list = []
    for item in data[8:]:
        originalValues.append(item)
        departTimeNum, duration = item.split(delimiter)
        time_or_date = departTimeNum
        if isDate:
            date = datetime.strptime(departTimeNum, '%d/%m/%Y')
            time_or_date = date.timetuple().tm_yday
        else:
            time = departTimeNum.split(':')
            time_or_date = int(time[0]) * 60 + int(time[1])
        departTimeNum_list.append(int(time_or_date))
        duration_list.append(int(duration))
    X = pd.DataFrame({'Departure Time num': departTimeNum_list, 'Duration': duration_list})

    # target = original_input['Price']
    # target = original_input['Price class']

    input_df = initialize(X, columns)
    # init_labels = init_labels_season_ltsa(input_df)
    init_labels = None
    hierarchies = create_hierarchies(input_df, columns, cluster_amounts, weights, cut_threshold, relax_steps,
                                        strat, best_max, functions, unique=unique,
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
