import argparse
import numpy as np


def ML(args: argparse.Namespace) -> np.ndarray:
    if args.dataset == 'DTINET':
        pass

    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['DTINET', 'E', 'GPCR', 'IC', 'NR'],
                        help='Data to train/test the model on')
    parser.add_argument('--sampling_ratio', type=str, choices=['1', '3', '5', '10', 'all'],
                        default='5', help='Ratio of negative sampling')  # baadan to code handle beshe k bayad int bashe
    parser.add_argument('--algorithm', type=str, choices=['kNN', 'RF', 'SVM', 'xgboost'],
                        default='RF', help='Machine learning algorithm')
    parser.add_argument('--kernel', type=str, choices=['linear', 'poly', 'rbf', 'sigmoid'],
                        default='rbf', help='Kernel type to be used in the SVM algorithm')
    parser.add_argument('--C', type=int, default=10,
                        help='Regularization parameter of the SVM algorithm')
    parser.add_argument('--num_estimators', type=int, default=100,
                        help='The number of decision trees in the random forest')
    parser.add_argument('--criterion', type=str, choices=['gini', 'entropy', 'log_loss'],
                        default='gini', help='The function to measure the quality of a split in the random forest algorithm')
    parser.add_argument('--max_features', type=str, choices=['sqrt', 'log2'],
                        default='sqrt', help='The number of features to consider when looking for the best split in the random forest algorithm')
    parser.add_argument('--k', type=int, default=5,
                        help='The number of neighbors considered for voting')
    parser.add_argument('--booster', type=str, choices=['gbtree', 'dart'],
                        default='gbtree', help='The model of xgboost')
    parser.add_argument('--comp_sim_mats', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to compute similarity matrices or not (set True at least once)')
    parser.add_argument('--log', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to log the optimization process')
    args = parser.parse_args()

    X_hat = ML(args)
    # np.savetxt('X_hat.txt', X_hat)
