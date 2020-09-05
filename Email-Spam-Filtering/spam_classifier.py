"""
This Program trains the classifier with the preclassified email dataset and classifies the new email.

To test the classifier:
1. Create a text file with spam/non spam email content from your personal mail box
2. Save the text file in the same directory of the program file.
3. Run the program spam_classifier.py and enter the file name without extention

    Example:
    >>> Enter the file name: test
4. The program traines the classifier and classifies the given test email and prints the output.
5. For testing the progam without training the classifier, run spam_classifier_pickle.py

"""
import nltk
import os
import random
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify
import pickle
#import matplotlib.pyplot as plt
 
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

# This function is used to train the classifier
def train_data(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    # initialise the training and test sets
    train_emailset, test_emailset = features[:train_size], features[train_size:]
    print ('Training set size = {0} emails'.format(len(train_emailset)))
    print ('Test set size = {0} emails'.format(len(test_emailset)))
    # train the classifier
    classifier = NaiveBayesClassifier.train(train_emailset)
    return train_emailset, test_emailset, classifier

# This function is used to evaluate the accuracy of the trainned 
def evaluate(train_set, test_set, classifier):
    # check how the classifier performs on the training and test sets
    accuracy_train=classify.accuracy(classifier, train_set)
    accuracy_test=classify.accuracy(classifier, test_set)
    print ('Accuracy on the training set = {0}%'.format(accuracy_train*100))
    print ('Accuracy of the test set = {0}%'.format(accuracy_test*100))
    # check which words are most informative for the classifier
    classifier.show_most_informative_features(100)
    return accuracy_test

# This function is used to store the classifier after reaching a threshold accuracy
def store_classifier(classifier):
    #Reference: https://pythonprogramming.net/pickle-classifier-save-nltk-tutorial/
    save_classifier = open("naivebayes.pickle","wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()
    

if __name__ == '__main__':

    # Import the email Data into workspace
    # Reference: https://blog.cambridgecoding.com/2016/01/25/implementing-your-own-spam-filter/

    # The below code imports the email content into the tuple and adds the label spam/ham
    
    new_email=input('Enter the file name: ') # Give the file name of text file with email content to test as input
    file_ext='.txt' # Adds the extention to the file name
    all_emails=[]
    spam_emails = load_data('enron_email/spam_email/')
    ham_emails = load_data('enron_email/ham_email/')
    for email in spam_emails:
        all_emails.append((email,'spam'))
    for email in ham_emails:
        all_emails.append((email,'ham'))# Stores all the emails with label in the tuple
    random.shuffle(all_emails) # The ham and spam emails  shuffled to mix both the emails randomly
    print ('The dataset contains: {0} emails'.format(len(all_emails)))
    
    # extract the features

    # The below code tokenizes the email content and removes the stop words.
    # The words after removing stop words and word frequency are the features extracted
    email_features = []
    for (email, label) in all_emails:
        email_features.append((get_features(email, ''), label))
    print ('Feature sets Extracted: {0}'.format(len(email_features)))

    # Visualize the extracted features
    #for (email, label) in all_emails:
    #    f_size=len(get_features(email, ''))
    #    if label == 'spam':
    #        plt.plot(i,f_size,'ro')
    #    else:
    #        plt.plot(i,f_size,'bo')
    #    plt.hold(True)
    #    i=i+1
    #plt.axis([0,5000,0,1000])
    #plt.xlabel('Emails')
    #plt.ylabel('Number of Features')
    #plt.title('Visualization of Extracted Features')
    #plt.show()
    
    # train the classifier

    # The below code is used to train the classifier
    # In the available dataset 80% is used for training and 20% is used for testing
    train_emailset, test_emailset, classifier = train_data(email_features, 0.8)

    # evaluate its performance
    accuracy_test=evaluate(train_emailset, test_emailset, classifier)

    # save the email in the classifier

    # The below code stores the classifier when the accuracy is more that 90%
    if accuracy_test>0.9:
       store_classifier(classifier)

    # Classifier for new email
    new_mail_dir=new_email+file_ext
    try:
        my_test=open(new_mail_dir).read()
    except valueerror:
        print('Please enter the valid file name')
    my_features=get_features(my_test,'')
    print('The content in the email is\n')
    print(my_test)
    res=classifier.classify(my_features)
    if res=='ham':
        print('\n\n The Email is not a Spam')

    else:
        print('\n\nThe Email is a spam')
