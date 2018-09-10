import json


def preprocess_data(filepath):
    """
    Using that function you are able to preprocess data and split training data from training labels
    :param filepath: unprocessed data
    :return: training data, training labels
    """
    train_labels = []
    train_data = []
    with open(filepath, encoding="utf8") as file:
        lines = file.readlines()[1:]
        for line in lines:
            json_formatted = json.loads(line)
            train_labels.append(json_formatted["topic"])
            train_data.append(json_formatted["question"] + " " + json_formatted["excerpt"])
        return train_labels, train_data
