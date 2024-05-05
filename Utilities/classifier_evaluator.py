import argparse
import csv
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def evaluate(
    ground_truth: list[int], predictions: list[int], prediction_scores: list[float]
) -> list[float]:
    accuracy = metrics.accuracy_score(ground_truth, predictions)
    balanced_accuracy = metrics.balanced_accuracy_score(ground_truth, predictions)
    precision = metrics.precision_score(ground_truth, predictions)
    recall = metrics.recall_score(ground_truth, predictions)
    specificity = metrics.recall_score(ground_truth, predictions, pos_label=0)
    f1_score = metrics.f1_score(ground_truth, predictions)
    MCC = metrics.matthews_corrcoef(ground_truth, predictions)
    AUC = metrics.roc_auc_score(ground_truth, prediction_scores)
    AUPR = metrics.average_precision_score(ground_truth, prediction_scores)

    return [accuracy, balanced_accuracy, precision, recall, specificity, f1_score, MCC, AUC, AUPR]


def threshold_tuning(ground_truth: list[int], prediction_scores: list[float], mode: str) -> int:
    if mode == "roc":
        fpr, tpr, thresholds = metrics.roc_curve(ground_truth, prediction_scores)
        gmeans = np.sqrt(tpr * (1 - fpr))
        optimal_threshold = thresholds[np.argmax(gmeans)]

    elif mode == "pr":
        pre, rec, thresholds = metrics.precision_recall_curve(ground_truth, prediction_scores)
        f1_scores = (2 * pre * rec) / (pre + rec)
        optimal_threshold = thresholds[np.argmax(f1_scores)]

    return optimal_threshold


def ROC_curve(ground_truth: list[int], prediction_scores: list[float]) -> None:
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth, prediction_scores)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(fpr, tpr, marker=".")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


def PR_curve(ground_truth: list[int], prediction_scores: list[float]) -> None:
    pre, rec, thresholds = metrics.precision_recall_curve(ground_truth, prediction_scores)
    perc_pos_samples = sum(ground_truth) / len(ground_truth)
    plt.plot([0, 1], [perc_pos_samples, perc_pos_samples], linestyle="--")
    plt.plot(rec, pre, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("y_path", type=str, help="Path to the ground truth file")
    parser.add_argument("y_hat_path", type=str, help="Path to the predictions file")
    parser.add_argument(
        "--probabilistic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether predictions are probabilistic",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["eval", "tht", "roc", "pr"],
        default="eval",
        help="Compute metrics, find optimal decision threshold, or draw a curve",
    )
    parser.add_argument(
        "--tht_mode",
        type=str,
        choices=["roc", "pr"],
        default="pr",
        help="Method of generating candidate thresholds while threshold tuning",
    )
    args = parser.parse_args()

    if args.y_path[-3:] == "txt":
        file = open(args.y_path)
        lines = [line for line in file]
        head = lines[0].split()
        tail = [line.split()[1:] for line in lines[1:]]
        y = np.array([list(map(int, row)) for row in tail]).flatten()
    else:
        file = open(args.y_path)
        csvreader = csv.reader(file)
        data = [row for row in csvreader]
        head = data[0]
        tail = data[1:]
        drugs = [row[0] for row in tail]
        y = np.array([list(map(int, row[1:])) for row in tail]).flatten()

    y_hat = np.loadtxt(args.y_hat_path).flatten()
    if args.probabilistic:
        threshold = threshold_tuning(y, y_hat, args.tht_mode)
        predictions = [0 if score < threshold else 1 for score in y_hat]

    if args.action == "eval":
        acc, bacc, pre, rec, spe, f1, MCC, AUC, AUPR = evaluate(y, predictions, y_hat)
        output_path = "/".join(args.y_hat_path.split("/")[:-1]) + "/"
        output_file = open(output_path + "evaluation_results.txt", "w")
        output_file.write("Accuracy: {}".format(acc))
        output_file.write("\nBalanced Accuracy: {}".format(bacc))
        output_file.write("\nPrecision: {}".format(pre))
        output_file.write("\nRecall: {}".format(rec))
        output_file.write("\nSpecificity: {}".format(spe))
        output_file.write("\nF1-score: {}".format(f1))
        output_file.write("\nMCC: {}".format(MCC))
        output_file.write("\nAUC: {}".format(AUC))
        output_file.write("\nAUPR: {}".format(AUPR))
        output_file.close()

    elif args.action == "tht":
        print(threshold)

    else:
        exec("curve = {}_curve".format(args.action.upper()))
        curve(y, y_hat)
