from sklearn.metrics import f1_score, accuracy_score
import torch


def evaluate(predictions, labels, threshold):
    norm = torch.where(predictions >= threshold, 1, 0)
    accuracy, f1 = accuracy_score(predictions, labels), f1_score(
        predictions, labels, average="micro"
    )

    lb_name = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "identity_attack",
        "insult",
        "threat",
    ]

    return accuracy, f1
