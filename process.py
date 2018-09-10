import json
import nltk

from nltk import RegexpTokenizer

try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')


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


def split_to_unigrams(train_data):
    """
    Using that function we can split the text (questions and excerpts) to unigrams
    :param train_data: train dataset
    :return: splitted texts
    """
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    processed_data = []
    for text in train_data:
        words = tokenizer.tokenize(text)
        processed_data.append(
            [word.lower() for word in words if
             word not in stop_words and word != "" and not word.isdigit()])
    return processed_data


labels, train_data = preprocess_data("training.json")
print(split_to_unigrams(train_data)[0])
