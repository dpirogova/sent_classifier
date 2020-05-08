# -*- coding: utf-8 -*-
import nltk
import random
from nltk.corpus import movie_reviews

# Hyperparameters

# the submission uses words as main features and bigrams for some support

# first idea was to use words as features
# it seems to be good, but it may fail while taking some expressions like "not good"
use_words = True
common_words_threshold = 2000 # overfitting begins at 2000

# using bigrams as features
# it may seem to be a good choice, but we need huge numbers of features to make it work well
use_bigrams = True
common_bigrams_threshold = 4000 # overfitting begins at 12000
 
# the results are slightly worse in comparison with using just words as features
use_char_ngrams = False
common_char_ngrams_threshold = 1000 # overfitting begins at 4000
n_in_ngrams = 8 # it is probably the best variant as 5,6,7 can show poorer results than ordinary words; 9 seems to overfit too fast

# prepare review data as a list of tuples:
# (list of tokens, category)
# category is positive / negative
review_data = [(list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]
random.seed(42) # as review_is sorted by categories, we need to shuffle it; random seed was chosen arbitrarily 
random.shuffle(review_data)
nr_reviews = len(review_data)

print("Total " + str(nr_reviews) + " movie reviews")

train_data = review_data[:int(0.8 * nr_reviews)] # first 80% of reviews
development_data = review_data[int(0.8 * nr_reviews):int(0.9 * nr_reviews)] # 80-90% of reviews
test_data = review_data[int(0.9 * nr_reviews):] # last 10% of reviews

if use_words:
    fd_words = nltk.FreqDist(word for (words, category) in train_data for word in words)
    top_words = [word for (word, freq) in fd_words.most_common(common_words_threshold)]

if use_bigrams:
    fd_bigrams = nltk.FreqDist( str(bigram)
            for (words, category) in train_data 
                for bigram in nltk.bigrams(words))
    top_bigrams = [bigram for (bigram, freq) in fd_bigrams.most_common(common_bigrams_threshold)]

if use_char_ngrams:
    fd_char_ngrams = nltk.FreqDist( str(ngram)
          for (words, category) in train_data
                for ngram in nltk.ngrams(" ".join(str(x) for x in words), n_in_ngrams, pad_left=True, left_pad_symbol=' '))
    top_char_ngrams = [ngram for (ngram, freq) in fd_char_ngrams.most_common(common_char_ngrams_threshold)]

def review_features(words_list):
    features = {}

    if use_words:
        for word in top_words:
            features['contains({})'.format(word)] = (word in words_list)

    if use_bigrams:
        review_bigrams = [str(bigram) for bigram in nltk.bigrams(words_list)]
        for bigram in top_bigrams:
            features['contains({})'.format(bigram)] = (bigram in review_bigrams)

    if use_char_ngrams:
        review_char_ngrams = [str(ngram) for ngram in nltk.ngrams(" ".join(str(x) for x in words_list), n_in_ngrams, pad_left=True, left_pad_symbol=' ')]
        for ngram in top_char_ngrams:
            features['contains({})'.format(ngram)] = (ngram in review_char_ngrams)
    return features

def reviews_to_featuresets(reviews):
    featuresets = []
    for (words_list, tag) in reviews:
        featuresets.append( (review_features(list(words_list)), tag) )
    return featuresets

train_featuresets = reviews_to_featuresets(train_data)
development_featuresets = reviews_to_featuresets(development_data)
test_featuresets = reviews_to_featuresets(test_data)
        
classifier = nltk.NaiveBayesClassifier.train(train_featuresets)

print(nltk.classify.accuracy(classifier, train_featuresets))
print(nltk.classify.accuracy(classifier, development_featuresets))
print(nltk.classify.accuracy(classifier, test_featuresets))

classifier.show_most_informative_features(200)

das_boot_series = nltk.word_tokenize('just watched the new das boot series - a so-called sequel to the well-known film. what can i say? i haven\'t seen such shizophrenical bullshit (yes, both storylines, about sailors and about gestapo) for a long time. i just can\'t tell which of the episodes is better - with lesbian sex in series four or with the ship (hold to your seat) with mixed german-romanian-ukranian crew, carrying jewish refugees to canada (and i remind you it is november 1942). hope that the second season will tell the story of what the writers use for imagination - probably, columbian cocaine from the special reichskanzelarie supplies, brought by u-666 special mission. basically, this is the bottom.');

print(classifier.classify(review_features(das_boot_series)))
