import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from itertools import chain
import matplotlib.pyplot as plt
import random


def average(iterable):
    size = 0
    sum = 0
    for num in iterable:
        sum += num
        size += 1
    return sum / size


def convert_boolean(value):
    if value == 'yes':
        return 1.0
    return 0.0


def split_input_and_output_data(frame, input_columns, output_columns):
    return (frame.drop(columns=output_columns), frame.drop(columns=input_columns))

class CvSplitter(object):
    def split(self, X, y, groups):
        dataset_length = X.shape[0]
        indices = [i for i in range(dataset_length)]
        for i in indices:
            random_index = random.randrange(0, dataset_length)
            indices[i], indices[random_index] = indices[random_index], indices[i]
        partition_index = dataset_length // 2
        yield indices[:partition_index], indices[partition_index:]
        yield indices[partition_index:], indices[:partition_index]

    def get_n_splits(self, X, y, groups):
        return 2


column_names = [
    'temperature', 'nausea', 'lumbar pain', 'urine pushing', 'micturition pains', 'urethra burning',
    'inflammation', 'nephritis'
]

converters = {prop: convert_boolean for prop in column_names if prop != 'temperature'}
converters['temperature'] = lambda x: float(x.replace(',', '.'))

csv_frame = pandas.read_csv(
    'diagnosis.data',
    delim_whitespace=True,
    names=column_names,
    usecols=column_names[:-1],
    encoding='UTF-16',
    converters=converters)

dataset, target = split_input_and_output_data(
    csv_frame, input_columns=csv_frame.columns[:-1], output_columns=csv_frame.columns[-1:])


def rank_features(dataset, target):
    ranking = SelectKBest(k='all').fit(dataset.values, target.values.ravel())
    scores = [score_with_label for score_with_label in zip(dataset.columns, ranking.scores_)]
    return sorted(scores, key=lambda x: x[1], reverse=True)


def cross_validate_5x2(dataset, target, classifier):
    scores = [
        x['test_score']
        for x in (cross_validate(classifier, dataset, target, cv=CvSplitter()) for _ in range(5))
    ]
    return average(chain(*scores))


def create_cross_validation_plot(dataset, target, classifier, features_list, filename, plot_title):
    def enumerate_datasets_with_features_increment(dataset, features_list):
        for i in range(len(features_list)):
            yield dataset.filter([label for label in features_list[:i + 1]])

    validation_results = [
        cross_validate_5x2(x.values, target.values.ravel(), classifier) * 100
        for x in enumerate_datasets_with_features_increment(dataset, features_list)
    ]
    plt.plot([_ + 1 for _ in range(len(validation_results))], validation_results)
    plt.title(plot_title)
    plt.xlabel('Ilość cech branych pod uwagę')
    plt.ylabel('Skuteczność w %')
    plt.ylim((0, 101))
    plt.grid(True)
    plt.savefig(filename)
    plt.clf()


ranking = rank_features(dataset, target)
print('Ranking cech:', ranking)
labels_sorted = [label for label, _ in ranking]

create_cross_validation_plot(
    dataset,
    target,
    NearestCentroid(),
    labels_sorted,
    filename='NM_euclidean',
    plot_title='Najmniejsza średnia (z metryką euklidesową)')

create_cross_validation_plot(
    dataset,
    target,
    NearestCentroid(metric='manhattan'),
    labels_sorted,
    filename='NM_manhattan',
    plot_title='Najmniejsza średnia (z metryką miejską)')

create_cross_validation_plot(
    dataset,
    target,
    KNeighborsClassifier(n_neighbors=1, metric='euclidean'),
    labels_sorted,
    filename='1NN_euclidean',
    plot_title='Najbliższy sąsiad (z metryką euklidesową)')

create_cross_validation_plot(
    dataset,
    target,
    KNeighborsClassifier(n_neighbors=1, metric='manhattan'),
    labels_sorted,
    filename='1NN_manhattan',
    plot_title='Najbliższy sąsiad (z metryką miejską)')

create_cross_validation_plot(
    dataset,
    target,
    KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
    labels_sorted,
    filename='5NN_euclidean',
    plot_title='k-najbliższych sąsiadów (z metryką euklidesową, k = 5)')

create_cross_validation_plot(
    dataset,
    target,
    KNeighborsClassifier(n_neighbors=5, metric='manhattan'),
    labels_sorted,
    filename='5NN_manhattan',
    plot_title='k-najbliższych sąsiadów (z metryką miejską, k = 5)')

create_cross_validation_plot(
    dataset,
    target,
    KNeighborsClassifier(n_neighbors=10, metric='euclidean'),
    labels_sorted,
    filename='10NN_euclidean',
    plot_title='k-najbliższych sąsiadów (z metryką euklidesową, k = 10)')

create_cross_validation_plot(
    dataset,
    target,
    KNeighborsClassifier(n_neighbors=10, metric='manhattan'),
    labels_sorted,
    filename='10NN_manhattan',
    plot_title='k-najbliższych sąsiadów (z metryką miejską, k = 10)')