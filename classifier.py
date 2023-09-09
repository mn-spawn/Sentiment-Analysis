# This file implements a Naive Bayes Classifier
import math

class BayesClassifier():
    """
    Naive Bayes Classifier
    file length: file length of training file
    sections: sections for incremental training
    """
    def __init__(self):
        self.postive_word_counts = {}
        self.negative_word_counts = {}
        self.percent_positive_sentences = 0
        self.percent_negative_sentences = 0
        self.file_length = 499
        self.file_sections = [self.file_length // 4, self.file_length // 3, self.file_length // 2]


    def train(self, train_data, train_labels, vocab):
        """
        This function builds the word counts and sentence percentages used for classify_text
        train_data: vectorized text
        train_labels: vectorized labels
        vocab: vocab from build_vocab
        """
        positive = 0
        negative = 0

        #init word dict
        for word in vocab:
            self.postive_word_counts.update({word: 1})
            self.negative_word_counts.update({word: 1})

        #loop through sentences
        for i, data in enumerate(train_data):
            if train_labels[i] == 1:
                positive += 1
                for j, word in enumerate(vocab):
                    if data[j] == 1:
                        self.postive_word_counts[word] +=1

            if train_labels[i] == 0:
                if train_labels[i] == 0:
                    negative += 1
                    for j, word in enumerate(vocab):
                        if data[j] == 1:
                            self.negative_word_counts[word] +=1


        self.percent_positive_sentences = positive/self.file_length
        self.percent_negative_sentences = negative/self.file_length

        return 1

    def classify_text(self, vectors, vocab):
        """
        vectors: [vector1, vector2, ...]
        predictions: [0, 1, ...]
        """

        predictions = []
        percent_pos = self.percent_positive_sentences
        percent_neg = self.percent_negative_sentences

        for vector in vectors:
            CL1 = 0
            CL0 = 0
            
            #class label 1
            numerator = 0
            for i in range(len(vector)):
                if vector[i] == 1:
                    if numerator == 0:
                        numerator = self.postive_word_counts[vocab[i]]
                    else:
                        numerator = (numerator * self.postive_word_counts[vocab[i]])

            CL1 = (numerator*percent_pos)


            #class label 2
            numerator = 0
            for i in range(len(vector)):
                if vector[i] == 1:
                    if numerator == 0:
                        numerator = self.negative_word_counts[vocab[i]]
                    else:
                        numerator = (numerator * self.negative_word_counts[vocab[i]])

            CL0 = (numerator*percent_neg)


            #choose max prob of both
            maxprob = max(CL1, CL0)
            if maxprob == CL1: 
                predictions.append(1)
        
            else: 
                predictions.append(0)

        return predictions
    