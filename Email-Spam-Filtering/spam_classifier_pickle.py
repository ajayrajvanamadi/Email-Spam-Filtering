"""
This Program trains the classifier with the preclassified email dataset and classifies the new email.

To test the classifier:
1. Create a text file with spam/non spam email content from your personal mail box
2. Save the text file in the same directory of the program file.
3. Run the program spam_classifier_pickle.py and enter the file name without extention

    Example:
    >>> Enter the file name: test
4. The program will use the pretainned classifier and classifies the given test email and prints the output.
5. For testing the progam after training the classifier once again, run spam_classifier.py
"""
import nltk
import os
import random
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify
import pickle
 
stop_words = stopwords.words('english')
 
# This function is used to load the email dataset in to the python workspace
# It uses file name as input and imports all the files in that folder
def load_data(folder_name):
    try:
        files_list = []
        file_list = os.listdir(folder_name)
        for a_file in file_list:
            file = open(folder_name + a_file,'r',encoding='cp437')
            files_list.append(file.read())
        file.close()
        return files_list
    except ValueError:
        print('Give a valid directorty of training dataset')

# This function is used for processing email content
# It uses text content of the email as input and tokenizes and lemmmatizes the text
def process_data(sentence):
    #Reference: http://stackoverflow.com/questions/20827741/nltk-naivebayesclassifier-training-for-sentiment-analysis
    #Reference: https://blog.cambridgecoding.com/2016/01/25/implementing-your-own-spam-filter/
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence)]

# This function is used to remove the stop words and count the word occurance of the processed data
def get_features(text, setting):
    # Reference: https://blog.cambridgecoding.com/2016/01/25/implementing-your-own-spam-filter/
    if setting=='bow':
        return {word: count for word, count in Counter(process_data(text)).items() if not word in stop_words}
    else:
        return {word: True for word in process_data(text) if not word in stop_words}

if __name__ == '__main__':
    new_email=input('Enter the file name: ') # Give the file name of text file with email content to test as input
    file_ext='.txt' # Adds the extention to the file name
    
    classifier_f = open("naivebayes.pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    # test new set
    new_mail_dir='/'+new_email+file_ext
    my_test=open(new_mail_dir).read()
    print(my_test)
    my_features=get_features(my_test,'')
    #print(my_features)
    res=classifier.classify(my_features)
    if res=='ham':
        print('\n\n The Email is not a Spam')

    else:
        print('\n\nThe Email is a spam')


    
