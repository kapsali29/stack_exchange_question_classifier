import json
import nltk
import re

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
        new_text = re.sub(r'http\S+', '', text)
        words = tokenizer.tokenize(new_text)
        processed_data.append(
            [word.lower() for word in words if
             word not in stop_words and word != "" and not word.isdigit() and word not in ['is', 'how', 'what', 'I']])
    return processed_data


def find_category_best_words(processed_data, train_labels):
    """
    Using that function you are able to find the words that contain the most amount of information for each category

    :param processed_data: unigrams
    :param train_labels: train data labels
    :return:
    """
    word_fd = nltk.FreqDist()
    label_word_fd = nltk.ConditionalFreqDist()
    word_scores = {}
    for i in range(len(train_labels)):
        text = processed_data[i]
        label = train_labels[i]
        for word in text:
            word_fd[word] += 1
            label_word_fd[label][word] += 1
    gis_word_count = label_word_fd['gis'].N()
    security_word_count = label_word_fd['security'].N()
    photo_word_count = label_word_fd['photo'].N()
    mathematica_word_count = label_word_fd['mathematica'].N()
    wordpress_word_count = label_word_fd["wordpress"].N()
    scifi_word_count = label_word_fd["scifi"].N()
    electronics_word_count = label_word_fd["electronics"].N()
    android_word_count = label_word_fd["android"].N()
    apple_word_count = label_word_fd["apple"].N()

    total_word_count = gis_word_count + security_word_count + \
                       photo_word_count + mathematica_word_count + wordpress_word_count + \
                       scifi_word_count + electronics_word_count + android_word_count + apple_word_count
    for word, freq in word_fd.items():
        gis_score = nltk.BigramAssocMeasures.chi_sq(label_word_fd['gis'][word],
                                                    (freq, gis_word_count), total_word_count)
        security_score = nltk.BigramAssocMeasures.chi_sq(label_word_fd['security'][word],
                                                         (freq, security_word_count), total_word_count)
        photo_score = nltk.BigramAssocMeasures.chi_sq(label_word_fd['photo'][word],
                                                      (freq, photo_word_count), total_word_count)
        mathematica_score = nltk.BigramAssocMeasures.chi_sq(label_word_fd['mathematica'][word],
                                                            (freq, mathematica_word_count), total_word_count)
        wordpress_score = nltk.BigramAssocMeasures.chi_sq(label_word_fd['wordpress'][word],
                                                          (freq, wordpress_word_count), total_word_count)
        electronics_score = nltk.BigramAssocMeasures.chi_sq(label_word_fd['electronics'][word],
                                                            (freq, electronics_word_count), total_word_count)
        android_score = nltk.BigramAssocMeasures.chi_sq(label_word_fd['android'][word],
                                                        (freq, android_word_count), total_word_count)
        apple_score = nltk.BigramAssocMeasures.chi_sq(label_word_fd['apple'][word],
                                                      (freq, apple_word_count), total_word_count)
        word_scores[
            word] = gis_score + security_score + photo_score + mathematica_score + wordpress_score + \
                    electronics_score + android_score + apple_score
        s = [(k, word_scores[k]) for k in sorted(word_scores, key=word_scores.get, reverse=True)][0:10000]
    return s

# labels, train_data = preprocess_data("training.json")
# pr_data = split_to_unigrams(train_data)
# res = find_category_best_words(pr_data, labels)
# print(res)
