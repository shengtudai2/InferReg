import random

def split_data(data_list, train_ratio=0.8):
    """
    Split the data_list into train_data and test_data according to the given train_ratio.

    Args:
        data_list (list): A list of PyTorch Geometric Data objects.
        train_ratio (float): The ratio of training data.

    Returns:
        train_data (list): A list of PyTorch Geometric Data objects for training.
        test_data (list): A list of PyTorch Geometric Data objects for testing.
    """
    # Shuffle data_list
    random.shuffle(data_list)

    # Calculate the number of training samples
    n_train = int(len(data_list) * train_ratio)

    # Split data_list into train_data and test_data
    train_data = data_list[:n_train]
    test_data = data_list[n_train:]

    return train_data, test_data

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics, such as accuracy, precision, recall, and F1-score.

    Args:
        y_true (list or tensor): A list or tensor of true labels.
        y_pred (list or tensor): A list or tensor of predicted labels.

    Returns:
        metrics (dict): A dictionary of evaluation metrics.
    """
    # Calculate evaluation metrics based on y_true and y_pred
    # ...

    metrics = {
        'accuracy': ...,
        'precision': ...,
        'recall': ...,
        'f1_score': ...
    }

    return metrics

def plot_results(results):
    """
    Plot the results, such as training loss, testing loss, and evaluation metrics.

    Args:
        results (dict): A dictionary of results to be plotted.
    """
    # Plot results, such as training loss, testing loss, and evaluation metrics
    # ...


def gpu_recall_score(labels, predictions):
    true_positives = ((predictions == 1) & (labels == 1)).sum().item()
    false_negatives = ((predictions == 0) & (labels == 1)).sum().item()
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0
    return recall

def gpu_precision_score(labels, predictions):
    true_positives = ((predictions == 1) & (labels == 1)).sum().item()
    false_positives = ((predictions == 1) & (labels == 0)).sum().item()
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0
    return precision

def gpu_f1_score(labels, predictions):
    precision = gpu_precision_score(labels, predictions)
    recall = gpu_recall_score(labels, predictions)
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    return f1

def gpu_auc_score(labels, probabilities):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels.cpu().numpy(), probabilities)

def gpu_accuracy_score(labels, predictions):
    correct = (predictions == labels).sum().item()
    total = labels.numel()
    acc = correct / total
    return acc