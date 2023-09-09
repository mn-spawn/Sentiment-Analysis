# CS331 Sentiment Analysis Assignment 3
# This file contains the processing functions
import re
from classifier import BayesClassifier
import csv
import matplotlib.pyplot as plt

def process_text(text):
    """
    Preprocesses the text: Remove apostrophes, punctuation marks, etc.
    Returns a list of text
    """

    preprocessed_text = []

    #strip extra white space and split into sentences
    text = text.rstrip()
    sentences = text.split('\n')
        
    #for each sentence, strip punctuation, replace tab and append    
    for sentence in sentences:
        stripped_sentence = re.sub(r'[^\w\s]', '', sentence)
        stripped_sentence = stripped_sentence.replace('\t', ' ')
        preprocessed_text.append(stripped_sentence.strip())
       
    return preprocessed_text

def build_vocab(preprocessed_text):
    """
    Builds the vocab from the preprocessed text
    preprocessed_text: output from process_text
    Returns unique text tokens
    """
    prevocab = []

    for sentence in preprocessed_text:
        #get the list of words
        sentence_words = sentence.split()

        #go through and set every word to it's lower case equivalent
        for i, word in enumerate(sentence_words):
                    prevocab.append(word.lower())

        #put it all into a unique alphabetical ordered list
        
    vocab = sorted(set(prevocab)) 
    vocab.remove('0')
    vocab.remove('1')
    return vocab

def vectorize_text(text, vocab):
    """
    Converts the text into vectors
    text: preprocess_text from process_text
    vocab: vocab from build_vocab
    Returns the vectorized text and the labels
    """

    vectorized_text = []
    classlabels = []

    for sentence in text:
        sentence_vector = []

        for word in vocab:
            if word in sentence:
                sentence_vector.append(1)
            else:
                sentence_vector.append(0)
        

        if '0' in sentence: classlabels.append(0)
        else: classlabels.append(1)
            

        vectorized_text.append(sentence_vector)

    return vectorized_text, classlabels
        
def accuracy(predicted_labels, true_labels):
    """
    predicted_labels: list of 0/1s predicted by classifier
    true_labels: list of 0/1s from text file
    return the accuracy of the predictions
    """
    total_sentences = len(predicted_labels)
    correct = 0
    for i in range(total_sentences):
        if predicted_labels[i] == true_labels[i]:
            correct += 1
             
    accuracy_score = correct/total_sentences       
    print(accuracy_score)       

    return accuracy_score

#=================================== Helper to write out to csv =====================#
def write_lists(data, file_path, headers, labels):
    with open(file_path, 'w', newline='') as file:
        headers.append("classlabel")

        for i,vector in enumerate(data):
            vector.append(labels[i])

        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data[0:])

#=====================================================================================#

def main():
    # Take in text files and outputs sentiment scores
    #setting up our BayesClass
    bc = BayesClassifier()

#========================================Training===================================#
    #read training file
    training = open("trainingSet.txt", "r")
    trainingtext = training.read()
    training.close()

    #processing the text here
    ptext = process_text(trainingtext)

    #only build vocab from training
    vocab = build_vocab(ptext)

    #vectorize all text
    train_data, train_labels = vectorize_text(ptext, vocab)
    #write_lists(train_data, "preprocessed_train.txt", vocab, train_labels)
    
    #training here
    bc.train(train_data, train_labels, vocab)

#======================================End Training===================================#

    # #read test file
    testData = open("testSet.txt", "r")
    test = testData.read()
    testData.close()

    # #processing
    ttext = process_text(test)

    # #combine training and test
    test_data, test_labels = vectorize_text(ttext, vocab)
    write_lists(test_data, "preprocessed_test.txt", vocab, test_labels)

    # #classifying here
    predictions = bc.classify_text(test_data, vocab)

    #write accuracy to results
    accuracy(predictions, test_labels)

    return 1


if __name__ == "__main__":
    main()


