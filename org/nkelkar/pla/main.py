__author__ = 'Nishant Kelkar'
import numpy as np
import sys


def weight_update(w, x, y):
    x = np.multiply(x, y)
    w += x
    return w


def get_failed_predictions(w, data_matrix, labels):
    m = data_matrix.shape[0]
    n = data_matrix.shape[1]  # get num cols

    if n != len(w):  # TODO: make this throw an error
        print 'Error! dim(weight vector) != dim(data feature space)'

    ret = {}

    for i in range(0, m):
        x_i = data_matrix[i, :]
        predicted_label = 0
        prediction = np.dot(x_i, w)
        if prediction < 0:
            predicted_label = -1
        elif prediction > 0:
            predicted_label = 1
        if predicted_label != labels[i]:
            ret[tuple(x_i)] = labels[i]

    return ret


def failed_predictions_single_pass(w, failed_predictions):
    # iterate over all failed predictions, update weight vector one-by-one
    for feature_vector, label in failed_predictions.iteritems():
        w = weight_update(w, np.asarray(feature_vector), label)

    return w


def stopping_criteria(w1, w2, epsilon):
    diff = np.subtract(w1, w2)
    if np.linalg.norm(diff) <= epsilon:
        return False
    return True


def main():
    data = open(sys.argv[1], 'r')

    num_records = int(sys.argv[2])
    num_features = int(sys.argv[3])
    threshold = int(sys.argv[4])

    data_matrix = np.zeros((num_records, num_features), dtype=float)
    labels = np.zeros((num_records,), dtype=int)
    w = np.zeros((num_features, ), dtype=float)

    i = 0
    for line in data:
        parts = line.split('\t')
        for j in range(0, num_features):
            data_matrix[i][j] = float(parts[j])
        labels[i] = parts[num_features]
        i += 1

    ret = get_failed_predictions(w, data_matrix, labels)

    iterations = 0
    while True:
        print len(ret)
        w = failed_predictions_single_pass(w, ret)
        iterations += 1
        if len(ret) == 0 or iterations > threshold:
            break
        ret = get_failed_predictions(w, data_matrix, labels)

    print w


if __name__ == "__main__":
    main()