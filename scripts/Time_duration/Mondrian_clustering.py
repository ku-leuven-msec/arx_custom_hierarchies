import itertools
import time
from collections import defaultdict
from datetime import datetime
import random
from enum import Enum, auto
from typing import Set, Callable
import numpy as np
import numpy.typing as npt
import pandas as pd


class DimSelectionStrategy(Enum):
    ENTROPY = auto()
    RELATIVE_RANGE = auto()
    UNIFORMITY = auto()
    MSE = auto()


class MondrianClustering:
    sizes: np.ndarray
    labels: np.ndarray
    ranges: np.ndarray
    save_all: bool
    max_levels: int
    max_labels: int
    cut_threshold: float
    auto_relax_steps: int
    dim_strategy: Callable
    __failed_label_dim: defaultdict[int, np.array]
    __failed_labels: Set[int]
    __auto_relax_decrement: float

    def __init__(self, sizes: npt.ArrayLike, cut_threshold: float = 0.0, auto_relax_steps: int = 0,
                 saveAll: bool = False, dim_strategy: DimSelectionStrategy = DimSelectionStrategy.RELATIVE_RANGE,
                 max_best: str = 'max'):
        if cut_threshold > 0.5 or cut_threshold < 0:
            raise ValueError(f'The cut-threshold must be between 0 and 0.5')
        if auto_relax_steps < 0:
            raise ValueError(f'The auto relax steps must be non-negative')
        if 1 in sizes:
            raise ValueError(f'The sizes cannot contain 1 as this is added by default')
        if max_best not in ['max', 'best']:
            raise ValueError(f'max_best must be "max" or "best". Not {max_best}')

        self.cut_threshold = cut_threshold
        self.auto_relax_steps = auto_relax_steps
        self.__auto_relax_decrement = cut_threshold / auto_relax_steps if auto_relax_steps != 0 else 0
        self.sizes = np.sort(np.array(sizes))
        self.save_all = saveAll
        self.max_labels = self.sizes[-1]
        self.max_levels = len(self.sizes) + 1 if not self.save_all else np.max(self.sizes) + 1
        self.max_best = max_best

        __dim_strategy_maps = {'max': {DimSelectionStrategy.RELATIVE_RANGE: self.__relative_range_priority,
                                       DimSelectionStrategy.UNIFORMITY: self.__uniformity_priority},
                               'best': {DimSelectionStrategy.UNIFORMITY: self.__best_uniformity_priority,
                                        DimSelectionStrategy.MSE: self.__best_mse_priority,
                                        DimSelectionStrategy.ENTROPY: self.__best_target_entropy_priority}}
        try:
            self.dim_strategy = __dim_strategy_maps[max_best][dim_strategy]
        except KeyError:
            raise NotImplementedError(f'The combination {max_best=} with {dim_strategy=} is not implemented')

    def __calculate_ranges(self, current_labels, X):
        ranges = np.full(shape=(self.max_labels, np.size(X, 1) * 2), fill_value=-1, dtype=int)
        merged = np.concatenate((X, np.reshape(current_labels, newshape=(len(current_labels), 1))), axis=1)
        m_sorted = merged[merged[:, -1].argsort()]
        grouped = np.split(m_sorted[:, 0:-1], np.unique(m_sorted[:, -1], return_index=True)[1][1:])
        for label, group in enumerate(grouped):
            ranges[label] = np.concatenate((np.min(group, axis=0), np.max(group, axis=0)), axis=0)
        return ranges

    @staticmethod
    def __relative_range_priority(X, possible_labels, current_labels, current_ranges, max_range, weights):
        total_dims = len(max_range)
        possible_ranges = current_ranges[possible_labels]
        tmp = [possible_ranges[:, :int(total_dims)], possible_ranges[:, int(total_dims):]]
        possible_sizes_normalized = (tmp[1] - tmp[0] + 1) / max_range
        possible_sizes_normalized *= weights

        sizes_ordered = np.unique(possible_sizes_normalized.flatten())[::-1]
        for size in sizes_ordered:
            rows, dims = np.where(possible_sizes_normalized == size)

            # randomize order with same values
            if len(rows) != 1:
                tmp = list(zip(rows, dims))
                random.shuffle(tmp)
                rows, dims = zip(*tmp)

            for row, dim in zip(rows, dims):
                label = possible_labels[row]
                yield label, dim

    @staticmethod
    def __uniformity_priority(X, possible_labels, current_labels, current_ranges, max_range, weights):
        total_dims = len(max_range)
        # trim X & current_labels to only include the ones in possible_labels
        tmp = np.isin(current_labels, possible_labels)
        X = X[tmp]
        current_labels = current_labels[tmp]
        possible_ranges = current_ranges[possible_labels]

        # calculate uniform probabilities for each label
        mins_maxs = [possible_ranges[:, :int(total_dims)], possible_ranges[:, int(total_dims):]]
        ranges = (mins_maxs[1] - mins_maxs[0] + 1)
        probs = 1 / ranges

        tvd_values = np.zeros(shape=ranges.shape, dtype=float)

        # group X by label
        merged = np.concatenate((X, np.reshape(current_labels, newshape=(len(current_labels), 1))), axis=1)
        m_sorted = merged[merged[:, -1].argsort()]
        grouped = np.split(m_sorted[:, 0:-1], np.unique(m_sorted[:, -1], return_index=True)[1][1:])

        for label_index in range(len(ranges)):
            for dim in range(total_dims):
                uniform_probs = np.full(shape=ranges[label_index][dim], fill_value=probs[label_index][dim])
                # calculate real probs
                data = grouped[label_index][:, dim]
                real_probs = np.zeros(shape=ranges[label_index][dim])
                unique, counts = np.unique(data.astype(int), return_counts=True)
                real_probs[unique - mins_maxs[0][label_index][dim]] = counts / len(data)
                # kl_divs[label_index][dim] = kl_div(real_probs, uniform_probs).sum() * weights[dim]
                tvd_values[label_index][dim] = np.abs(uniform_probs - real_probs).sum() / 2 * weights[dim]

        tvd_ordered = np.unique(tvd_values.flatten())[::-1]
        for tvd in tvd_ordered:
            rows, dims = np.where(tvd_values == tvd)

            # randomize order with same values
            if len(rows) != 1:
                tmp = list(zip(rows, dims))
                random.shuffle(tmp)
                rows, dims = zip(*tmp)

            for row, dim in zip(rows, dims):
                label = possible_labels[row]
                yield label, dim

    @staticmethod
    def __best_uniformity_priority(X, y, current_best_score, dim, label, median_filter, weights, current_ranges):

        current_best_score = -1 if current_best_score is None else current_best_score

        # calculate increase in uniformity when applying this cut
        def calculate_tvd(value_range, values):
            range_size = value_range[1] - value_range[0] + 1
            uniform_probs = np.full(shape=range_size, fill_value=1 / range_size)
            # calculate real probs
            real_probs = np.zeros(shape=range_size)
            unique, counts = np.unique(values.astype(int), return_counts=True)
            real_probs[unique - value_range[0]] = counts / len(values)
            return np.abs(uniform_probs - real_probs).sum() / 2 * weights[dim]

        data_part0 = X[~median_filter]
        data_part1 = X[median_filter]
        old_ranges = current_ranges[label]
        ranges_part0 = np.concatenate((data_part0.min(axis=0), data_part0.max(axis=0)))
        ranges_part1 = np.concatenate((data_part1.min(axis=0), data_part1.max(axis=0)))

        old_range = [old_ranges[dim], old_ranges[int(len(old_ranges) / 2 + dim)]]
        range_part0 = [ranges_part0[dim], ranges_part0[int(len(ranges_part0) / 2 + dim)]]
        range_part1 = [ranges_part1[dim], ranges_part1[int(len(ranges_part1) / 2 + dim)]]
        old_data = X[:, dim]
        data0 = data_part0[:, dim]
        data1 = data_part1[:, dim]

        old_tvd = calculate_tvd(old_range, old_data)
        tvd0 = calculate_tvd(range_part0, data0)
        tvd1 = calculate_tvd(range_part1, data1)
        decrease = old_tvd - (tvd0 * len(data_part0) + tvd1 * len(data_part1)) / len(old_data)

        if decrease > current_best_score:
            return decrease, True
        else:
            return current_best_score, False

    @staticmethod
    def __best_mse_priority(X, y, current_best_score, dim, label, median_filter, weights, current_ranges):
        # calculate lowest mse of the target (see infogain mondrian paper)
        current_best_score = np.inf if current_best_score is None else current_best_score

        if y is None:
            raise ValueError('A y array must be given when MSE is used')

        y_part0 = y[~median_filter]
        y_part1 = y[median_filter]

        error = np.sum(np.power(y_part0 - np.mean(y_part0), 2)) + np.sum(np.power(y_part1 - np.mean(y_part1), 2))
        error /= len(y)
        error = error * weights[dim]

        if error < current_best_score:
            return error, True
        else:
            return current_best_score, False

    @staticmethod
    def __best_target_entropy_priority(X, y, current_best_score, dim, label, median_filter, weights, current_ranges):
        # calculate lowest target weighted entropy (see infogain mondrian paper)
        current_best_score = np.inf if current_best_score is None else current_best_score

        if y is None:
            raise ValueError('A y array must be given when target entropy is used')

        y_part0 = y[~median_filter]
        y_part1 = y[median_filter]

        probs0 = np.unique(y_part0, return_counts=True)[1] / len(y_part0)
        probs1 = np.unique(y_part1, return_counts=True)[1] / len(y_part1)

        entropy0 = -probs0*np.log(probs0)
        entropy1 = -probs1*np.log(probs1)

        entropy = (len(y_part0) * entropy0.sum() + len(y_part1) * entropy1.sum()) / len(y)

        entropy = entropy * weights[dim]

        if entropy < current_best_score:
            return entropy, True
        else:
            return current_best_score, False

    def __get_best_split_label(self, current_labels, current_ranges, max_range, X, weights, y) -> (
            tuple[int, int, int, np.array, np.array] | tuple[None, None, None, None, None]):

        counts = np.bincount(current_labels)
        counts[list(self.__failed_labels)] = 0

        # todo remove
        # tmp fix to not prioritize sizes
        # mask = np.zeros(counts.shape, dtype=bool)
        # mask[list(self.__failed_labels)] = True
        # counts[~mask] = 1

        for max_counts in np.unique(counts)[::-1]:

            if not max_counts:
                continue

            # possible_labels = unique_labels[counts == max_counts]
            possible_labels = np.array((counts == max_counts).nonzero()[0])

            current_best = (None, None, None, None)
            current_best_score = None

            for label, dim in itertools.product(possible_labels, range(X.shape[1])):
                if self.__failed_label_dim[label][dim]:
                    continue

                current_labels_filter = current_labels == label
                cluster_data = X[current_labels_filter].astype(int)
                cluster_target_data = y[current_labels_filter] if y is not None else None

                median = int(np.median(cluster_data[:, dim]))

                # test 2 situations, one where median values are in lower square, another where they are in the
                # higher square. Usefull when data contains many values having the same value as the median.
                failed_medians_count = 0
                for current_median in reversed(range(median - 1, median + 1)):
                    median_filter = (cluster_data[:, dim] > current_median)

                    part1_percentage = np.mean(median_filter)
                    if part1_percentage > self.cut_threshold and 1 - part1_percentage > self.cut_threshold:

                        current_best_score, improved = self.dim_strategy(cluster_data, cluster_target_data,
                                                                         current_best_score, dim, label, median_filter,
                                                                         weights, current_ranges)

                        if improved:
                            current_best = (label, dim, current_median, current_labels_filter)
                    else:
                        failed_medians_count += 1
                if failed_medians_count == 2:
                    self.__failed_label_dim[label][dim] = True

                if self.__failed_label_dim[label].sum() == X.shape[1]:
                    self.__failed_labels.add(label)

            if current_best[0] is not None:
                del self.__failed_label_dim[current_best[0]]
                return current_best + ((X[:, current_best[1]] > current_best[2]),)

        print('NO VALID CUT FOUND')
        return None, None, None, None, None

    def __get_split_label(self, current_labels, current_ranges, max_range, X, weights, y) -> (
            tuple[int, int, int, np.array, np.array] | tuple[None, None, None, None, None]):
        counts = np.bincount(current_labels)
        counts[list(self.__failed_labels)] = 0

        # todo remove
        # tmp fix to not prioritize sizes
        # mask = np.zeros(counts.shape, dtype=bool)
        # mask[list(self.__failed_labels)] = True
        # counts[~mask] = 1

        for max_counts in np.unique(counts)[::-1]:

            if not max_counts:
                continue

            # possible_labels = unique_labels[counts == max_counts]
            possible_labels = np.array((counts == max_counts).nonzero()[0])

            for label, dim in self.dim_strategy(X, possible_labels, current_labels, current_ranges, max_range, weights):
                if self.__failed_label_dim[label][dim]:
                    continue

                current_labels_filter = current_labels == label
                tmp = X[current_labels_filter]
                median = int(np.median(tmp[:, dim]))

                # test 2 situations, one where median values are in lower square, another where they are in the
                # higher square. Usefull when data contains many values having the same value as the median.
                for current_median in reversed(range(median - 1, median + 1)):
                    data_filtered = tmp[tmp[:, dim] > current_median]
                    part1_percentage = len(data_filtered) / len(tmp)
                    if part1_percentage > self.cut_threshold and 1 - part1_percentage > self.cut_threshold:
                        median_filter = (X[:, dim] > current_median)
                        del self.__failed_label_dim[label]
                        return label, dim, current_median, current_labels_filter, median_filter

                self.__failed_label_dim[label][dim] = True

                if self.__failed_label_dim[label].sum() == X.shape[1]:
                    self.__failed_labels.add(label)

        print('NO VALID CUT FOUND')
        return None, None, None, None, None

    # @profile
    def fit_predict(self, X: npt.ArrayLike, y: npt.ArrayLike = None, weights: npt.ArrayLike = None,
                    init_labels: npt.ArrayLike = None, progress: bool = False):
        X = np.array(X)
        y = np.array(y) if y is not None else None

        self.__failed_label_dim = defaultdict(lambda: np.zeros(X.shape[1], dtype=bool))
        self.__failed_labels = set()

        if weights is None:
            weights = np.ones(X.shape[1])
        else:
            weights = np.array(weights)

        if len(weights) != X.shape[1]:
            raise ValueError(
                f'The given weights must have the same length as the dataset columns. Expected {X.shape[1]} got {len(weights)}')

        # validate that X has enough unique points to create the amount of groups
        if self.max_labels > (uniques := len(np.unique(X, axis=0))):
            raise ValueError(f'The given dataset needs {self.max_labels} unique rows but has {uniques}')

        if init_labels is not None:
            init_labels = np.array(init_labels)
            init_labels = np.unique(init_labels, return_inverse=True)[1]
            if (a := np.max(init_labels) + 1) >= self.sizes[0]:
                raise ValueError(
                    f'There are more unique initial labels than the smallest cluster to save. Expected <{self.sizes[0]} got {a}')
            if init_labels.shape != (len(X),):
                raise ValueError(f'The init labels must have shape {(len(X),)} got {init_labels.shape}')

        # output variables
        self.labels = np.full(shape=(self.max_levels, len(X)), fill_value=-1, dtype=int)
        self.ranges = np.full(shape=(self.max_levels, self.max_labels, np.size(X, 1) * 2), fill_value=-1, dtype=int)

        # variables that will change iteratively
        current_labels = np.full(np.size(X, 0), fill_value=-1, dtype=int)
        current_ranges = np.full(shape=(self.max_labels, np.size(X, 1) * 2), fill_value=-1, dtype=int)

        # split until we have max sizes of cells
        current_level = 0

        pbar = None
        if progress:
            from tqdm import tqdm
            pbar = tqdm(total=self.max_labels)
        try:
            if init_labels is not None:
                # initialize the init labels, the init value replaces the suppression level
                current_labels[:] = init_labels
                current_ranges[:] = self.__calculate_ranges(current_labels, X)
            else:
                # suppression level
                current_labels[:] = 0
                current_ranges[0] = np.concatenate((np.min(X, axis=0), np.max(X, axis=0)), axis=0)

            # calculate size of dataset dimensions needed for normalization
            max_range = np.concatenate((np.min(X, axis=0), np.max(X, axis=0)), axis=0)
            tmp = np.split(max_range, 2)
            max_range = tmp[1] - tmp[0] + 1

            # save the data of the first level
            self.labels[current_level] = current_labels
            self.ranges[current_level] = current_ranges
            current_level += 1
            start_label = np.max(current_labels) + 1
            if pbar is not None:
                pbar.update(start_label)

            for label in range(start_label, self.max_labels):
                if self.max_best == 'max':
                    split_label, split_dim, median, current_label_filer, median_filter = self.__get_split_label(
                        current_labels, current_ranges, max_range, X, weights, y)
                else:
                    split_label, split_dim, median, current_label_filer, median_filter = self.__get_best_split_label(
                        current_labels, current_ranges, max_range, X, weights, y)

                if split_label is None:
                    if self.auto_relax_steps != 0:
                        # automatic decrementing if the cut-off is enabled
                        while split_label is None and self.cut_threshold != 0:
                            new_threshold = self.cut_threshold - self.__auto_relax_decrement
                            new_threshold = 0 if new_threshold < 0 else new_threshold
                            print(f'Relaxing the cut threshold from {self.cut_threshold} to {new_threshold}')
                            self.cut_threshold = new_threshold
                            # reset failed labels and dims
                            self.__failed_label_dim = defaultdict(lambda: defaultdict(lambda: False))
                            self.__failed_labels = set()
                            if self.max_best == 'max':
                                split_label, split_dim, median, current_label_filer, median_filter = self.__get_split_label(
                                    current_labels, current_ranges, max_range, X, weights, y)
                            else:
                                split_label, split_dim, median, current_label_filer, median_filter = self.__get_best_split_label(
                                    current_labels, current_ranges, max_range, X, weights, y)

                    if split_label is None:
                        print(
                            f'Early stopping as no allowed cuts are found. Ended with {np.max(current_labels)} clusters while {self.max_labels} where requested.')
                        self.labels[current_level] = current_labels
                        self.ranges[current_level] = current_ranges
                        return

                # split range
                # recalculate ranges in all dims
                data_part0 = X[current_label_filer & ~median_filter]
                data_part1 = X[current_label_filer & median_filter]

                new_range_0 = np.concatenate((data_part0.min(axis=0), data_part0.max(axis=0)))
                new_range_1 = np.concatenate((data_part1.min(axis=0), data_part1.max(axis=0)))

                current_ranges[split_label] = new_range_0
                current_ranges[label] = new_range_1

                current_labels[current_label_filer & median_filter] = label

                if pbar is not None:
                    pbar.update(1)

                if self.save_all or label + 1 in self.sizes:
                    # save the data of this level
                    self.labels[current_level] = current_labels
                    self.ranges[current_level] = current_ranges
                    current_level += 1
        except Exception as e:
            raise e
        finally:
            if pbar is not None:
                pbar.close()


def initialize(input_df, labels, is_dates, ref_day):
    data = input_df.copy()[labels]

    for label, is_date in zip(labels, is_dates):
        col = data[label]
        if is_date:
            data[f'{label} ml'] = col.map(lambda x: (datetime.fromisoformat(x) - ref_day).days)
        else:
            data[f'{label} ml'] = col

    return data


def draw_matplotlib(data, ranges):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    tmp = np.split(ranges, 2, axis=2)
    lengths = tmp[1] - tmp[0]

    for current_lengths, current_ranges in zip(lengths, ranges):
        plt.scatter(data[labels[0]], data[labels[1]], s=0.1)

        for current_length, current_range in zip(current_lengths, current_ranges):
            if -1.0 not in current_range:
                plt.gca().add_patch(
                    Rectangle((current_range[0], current_range[1]), current_length[0], current_length[1],
                              edgecolor='red',
                              facecolor='none',
                              lw=1))
        plt.show(dpi=600)


def draw_plotly(data, ranges, labels):
    import plotly.graph_objects as go

    for level, current_ranges in enumerate(ranges):
        # fig = go.Figure(data=[go.Scatter(x=data[labels[0]],y=data[labels[1]],mode='markers',)])
        fig = go.Figure()

        colors = list(get_colors(len(current_ranges)))
        fig.update_xaxes(range=[data[labels[0]].min(), data[labels[0]].max()], showgrid=False)
        fig.update_yaxes(range=[data[labels[1]].min(), data[labels[1]].max()])

        for index, current_range in enumerate(current_ranges):
            if -1.0 not in current_range:
                trace = go.Scatter(
                    x=[current_range[0], current_range[0], current_range[2], current_range[2], current_range[0]],
                    y=[current_range[1], current_range[3], current_range[3], current_range[1], current_range[1]],
                    line={'color': colors[index]}, name=str(current_range),
                    fill='toself', hoverlabel={'namelength': -1})
                fig.add_trace(trace)

        fig.update_shapes(dict(xref='x', yref='y'))
        fig.write_html(f'{output_path}{name}_{level}.html')


def get_random_color(colors):
    color = list(np.random.choice(range(256), size=3))
    color = '#%02x%02x%02x' % (color[0], color[1], color[2])
    while color in colors:
        color = list(np.random.choice(range(256), size=3))
        color = '#%02x%02x%02x' % (color[0], color[1], color[2])
    colors.add(color)
    return color


def get_colors(num_colors):
    colors = set()
    while len(colors) != num_colors:
        get_random_color(colors)
    return colors


if __name__ == '__main__':
    input_path = '/mnt/EncryptedTank/encrypted/Abs-Politie/scratch/absences_merged.csv'
    cluster_amounts = [500, 250, 100, 50, 20, 5]
    cluster_amounts = [10000, 5000, 2000, 1000, 500, 250, 100, 50, 20, 5]
    # cluster_amounts = [5]
    ref_day = datetime(year=2018, month=12, day=31)
    labels = ['Start Date', 'Time']
    is_dates = [True, False]
    weights = [1.0, 1.0]
    cut_threshold = 0
    output_path = '../output/drawings/mondrian/'
    name = 'no_weights_0.4'
    # test
    original_input = pd.read_csv(input_path, sep=';', decimal=',')
    original_input = original_input[0 < original_input["Time"]]
    original_input = original_input[original_input["Time"] < 365]
    input_df = initialize(original_input, labels, is_dates, ref_day)

    ml_labels = [f'{label} ml' for label in labels]
    data_to_cluster = input_df[ml_labels]

    # random generate 4 init labels
    np.random.seed(420)
    # init_labels = np.random.randint(4, size=(len(data_to_cluster),))
    init_labels = None

    # only first 5000 datapoints
    # input_df = input_df.iloc[:5000]
    start = time.time()
    print('Starting clustering')
    model = MondrianClustering(cluster_amounts, cut_threshold=cut_threshold,
                               dim_strategy=DimSelectionStrategy.UNIFORMITY)
    model.fit_predict(data_to_cluster, init_labels=init_labels, weights=weights, progress=True)
    print(f'Ended clustering in {time.time() - start} seconds')

    # draw_plotly(data_to_cluster, model.ranges, ml_labels)
