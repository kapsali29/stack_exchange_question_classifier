from process import preprocess_data, split_to_unigrams, find_category_best_words, train_classifier, clean_test_data, \
    read_test_labels, _predict

labels, train_data = preprocess_data("training.json")
pr_data = split_to_unigrams(train_data)
res = find_category_best_words(pr_data, labels)
clf, features = train_classifier(res, pr_data, labels)
test_data = clean_test_data("input00.txt")
test_labels = read_test_labels("output00.txt")
print(_predict(test_data, test_labels, clf, features))