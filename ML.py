import os
import argparse
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from Utilities import (
    compute_similarity_matrices,
    compute_embeddings,
    compute_ratioed_data,
    classifier_evaluator,
)


def ML(args: argparse.Namespace) -> None:
    if args.comp_sim_mats and args.dataset == "DTINET":
        compute_similarity_matrices.save_to_file()
    if args.comp_embs:
        compute_embeddings.save_to_file(args.dataset)
    if args.comp_ratioed_data:
        compute_ratioed_data.save_to_file(args.dataset, args.ratio)

    path = "Data/DTINet dataset/" if args.dataset == "DTINET" else "Data/Gold standard dataset/"
    X = np.load(path + args.dataset + "_" + args.ratio + "-to-1_X.npz")["arr_0"]
    y = np.load(path + args.dataset + "_" + args.ratio + "-to-1_y.npz")["arr_0"]

    kfolder = StratifiedKFold(n_splits=args.num_folds, shuffle=True)
    folds = kfolder.split(X, y)

    alg_map = {
        "kNN": KNeighborsClassifier,
        "RF": RandomForestClassifier,
        "SVM": SVC,
        "xgboost": XGBClassifier,
    }
    vars_map = {
        "kNN": {"n_neighbors": args.k},
        "RF": {
            "n_estimators": args.num_estimators,
            "criterion": args.criterion,
            "max_features": args.max_features,
        },
        "SVM": {
            "C": args.C,
            "kernel": args.kernel,
            "probability": True,
            "class_weight": "balanced",
        },
        "xgboost": {
            "objective": "binary:logistic",
            "booster": args.booster,
            "n_estimators": args.num_estimators,
        },
    }

    if args.oa_eval:
        overall_eval_file = open(
            "Results/ML/{}_{}_{}-to-1_overall evaluation.txt".format(
                args.algorithm, args.dataset, args.ratio
            ),
            "w",
        )
        metric_lists = [[], [], [], [], [], [], [], [], []]
    fold_no = 1
    for train_indices, test_indices in folds:
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        classifier = alg_map[args.algorithm](**vars_map[args.algorithm])
        classifier.fit(X_train, y_train)
        y_hat = classifier.predict(X_test)
        class_probs = classifier.predict_proba(X_test)
        pos_class_probs = np.array([prob[1] for prob in class_probs])

        if args.oa_eval:
            metrics = classifier_evaluator.evaluate(y_test, y_hat, pos_class_probs)
            for i in range(len(metric_lists)):
                metric_lists[i].append(metrics[i])

        np.savetxt(
            "Results/ML/{}_{}_{}-to-1_fold{}_y.txt".format(
                args.algorithm, args.dataset, args.ratio, fold_no
            ),
            y_test,
        )
        np.savetxt(
            "Results/ML/{}_{}_{}-to-1_fold{}_y-hat.txt".format(
                args.algorithm, args.dataset, args.ratio, fold_no
            ),
            y_hat,
        )
        np.savetxt(
            "Results/ML/{}_{}_{}-to-1_fold{}_probabilistic-y-hat.txt".format(
                args.algorithm, args.dataset, args.ratio, fold_no
            ),
            pos_class_probs,
        )

        fold_no += 1

    if args.oa_eval:
        overall_eval_file.write(
            "Accuracy => AVG : {}  STDEV : {}".format(
                np.average(metric_lists[0]), np.std(metric_lists[0])
            )
        )
        overall_eval_file.write(
            "\nBalanced accuracy => AVG : {}  STDEV : {}".format(
                np.average(metric_lists[1]), np.std(metric_lists[1])
            )
        )
        overall_eval_file.write(
            "\nPrecision => AVG : {}  STDEV : {}".format(
                np.average(metric_lists[2]), np.std(metric_lists[2])
            )
        )
        overall_eval_file.write(
            "\nRecall => AVG : {}  STDEV : {}".format(
                np.average(metric_lists[3]), np.std(metric_lists[3])
            )
        )
        overall_eval_file.write(
            "\nSpecificity => AVG : {}  STDEV : {}".format(
                np.average(metric_lists[4]), np.std(metric_lists[4])
            )
        )
        overall_eval_file.write(
            "\nF1-score => AVG : {}  STDEV : {}".format(
                np.average(metric_lists[5]), np.std(metric_lists[5])
            )
        )
        overall_eval_file.write(
            "\nMCC => AVG : {}  STDEV : {}".format(
                np.average(metric_lists[6]), np.std(metric_lists[6])
            )
        )
        overall_eval_file.write(
            "\nAUC => AVG : {}  STDEV : {}".format(
                np.average(metric_lists[7]), np.std(metric_lists[7])
            )
        )
        overall_eval_file.write(
            "\nAUPR => AVG : {}  STDEV : {}".format(
                np.average(metric_lists[8]), np.std(metric_lists[8])
            )
        )
        overall_eval_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        choices=["DTINET", "E", "GPCR", "IC", "NR"],
        help="Data to train/test the model on",
    )
    parser.add_argument(
        "--ratio",
        type=str,
        choices=["1", "3", "5", "10", "all"],
        default="5",
        help="Ratio of negative sampling",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=10,
        help="Number of folds used in the stratified K-Fold cross-validator",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["kNN", "RF", "SVM", "xgboost"],
        default="RF",
        help="Choice of machine learning algorithm",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        choices=["linear", "poly", "rbf", "sigmoid"],
        default="rbf",
        help="Kernel type to be used in the SVM algorithm",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=10,
        help="Regularization parameter of the SVM algorithm",
    )
    parser.add_argument(
        "--num_estimators",
        type=int,
        default=100,
        help="The number of decision trees in the random forest",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        choices=["gini", "entropy", "log_loss"],
        default="gini",
        help="The function to measure the quality of a split in the random forest algorithm",
    )
    parser.add_argument(
        "--max_features",
        type=str,
        choices=["sqrt", "log2"],
        default="sqrt",
        help="The number of features to consider when looking for the best split in the random forest algorithm",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="The number of neighbors considered for voting",
    )
    parser.add_argument(
        "--booster",
        type=str,
        choices=["gbtree", "dart"],
        default="gbtree",
        help="The model of xgboost",
    )
    parser.add_argument(
        "--comp_sim_mats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to compute the similarity matrices or not (set True at least once for the )",
    )
    parser.add_argument(
        "--comp_embs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to compute the embeddings or not (set True at least once)",
    )
    parser.add_argument(
        "--comp_ratioed_data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to compute the ratioed dataset or not (set True at least once for each pair of dataset and ratio)",
    )
    parser.add_argument(
        "--oa_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to evaluate the performance of the model on average or not",
    )
    args = parser.parse_args()

    if not os.path.exists("Results"):
        os.mkdir("Results")
        os.mkdir("Results/ML")
    elif os.path.exists("Results") and not os.path.exists("Results/ML"):
        os.mkdir("Results/ML")

    ML(args)
