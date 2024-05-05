import csv
import numpy as np
from sklearn.utils import shuffle


def save_to_file(dataset: str, ratio: str) -> None:
    if dataset == "DTINET":
        file = open("Data/DTINet dataset/mat_drug_protein.csv")
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        X = np.array([list(map(int, row[1:])) for row in tail])
        D = np.loadtxt("Data/DTINet dataset/D.txt")
        T = np.loadtxt("Data/DTINet dataset/T.txt")

        y = []
        X_indices = []
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                y.append(X[i, j])
                X_indices.append([i, j])
        X_indices, y = shuffle(X_indices, y)

        if ratio == "all":
            shuffled_X = []
            for tup in X_indices:
                shuffled_X.append(np.concatenate((D[tup[0]], T[tup[1]])))
            X = shuffled_X
        else:
            ones = [X_indices[i] for i in range(len(X_indices)) if y[i] == 1]
            zeros = [X_indices[i] for i in range(len(X_indices)) if y[i] == 0]
            ratioed_zeros = [index for index in zeros[: eval(ratio) * len(ones)]]
            ratioed_X_indices = ones + ratioed_zeros
            ratioed_y = [1 for _ in range(len(ones))] + [0 for _ in range(len(ratioed_zeros))]
            ratioed_X_indices, ratioed_y = shuffle(ratioed_X_indices, ratioed_y)

            ratioed_X = []
            for tup in ratioed_X_indices:
                ratioed_X.append(np.concatenate((D[tup[0]], T[tup[1]])))
            X = ratioed_X
            y = ratioed_y

        np.savez_compressed(
            "Data/DTINet dataset/" + dataset + "_" + ratio + "-to-1_y.npz",
            np.array(y),
        )
        np.savez_compressed(
            "Data/DTINet dataset/" + dataset + "_" + ratio + "-to-1_X.npz",
            np.array(X),
        )

    else:
        file = open("Data/Gold standard dataset/" + dataset.lower() + "_admat_dgc.txt")
        lines = [line for line in file]
        head = lines[0].split()
        tail = [line.split()[1:] for line in lines[1:]]
        X = np.array([list(map(int, row)) for row in tail])
        T = np.loadtxt("Data/Gold standard dataset/T.txt")
        D = np.loadtxt("Data/Gold standard dataset/D.txt")

        y = []
        X_indices = []
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                y.append(X[i, j])
                X_indices.append([i, j])
        X_indices, y = shuffle(X_indices, y)

        if ratio == "all":
            shuffled_X = []
            for tup in X_indices:
                shuffled_X.append(np.concatenate((T[tup[0]], D[tup[1]])))
            X = shuffled_X
        else:
            ones = [X_indices[i] for i in range(len(X_indices)) if y[i] == 1]
            zeros = [X_indices[i] for i in range(len(X_indices)) if y[i] == 0]
            ratioed_zeros = [index for index in zeros[: eval(ratio) * len(ones)]]
            ratioed_X_indices = ones + ratioed_zeros
            ratioed_y = [1 for _ in range(len(ones))] + [0 for _ in range(len(ratioed_zeros))]
            ratioed_X_indices, ratioed_y = shuffle(ratioed_X_indices, ratioed_y)

            ratioed_X = []
            for tup in ratioed_X_indices:
                ratioed_X.append(np.concatenate((T[tup[0]], D[tup[1]])))
            X = ratioed_X
            y = ratioed_y

        np.savez_compressed(
            "Data/Gold standard dataset/" + dataset + "_" + ratio + "-to-1_y.npz",
            np.array(y),
        )
        np.savez_compressed(
            "Data/Gold standard dataset/" + dataset + "_" + ratio + "-to-1_X.npz",
            np.array(X),
        )
