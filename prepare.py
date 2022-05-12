import unicodedata
import re
import json
import pandas as pd

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

def basic_clean(string):
    """
    This function will take in a string and perform basic cleaning procedutes. It will convert all characters
    to lower case, remove accented characters using unicode, and remove all special character 
    and symbols that are not alphanumeric characters.
    """
    
    #Convert to lower case
    string = string.lower()
    
    #Normalize and remove inconsistencies, 
    #encode into ascii byte strings and ignore unknown chars,
    #decode back into a UTF-8 string
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('UTF-8')
    
    #Use regex to replace remove/replace all special characters
    string = re.sub(r"[^a-z0-9\s']", '', string)
    
    return string

def tokenize(string):
    """
    This function will take in a string, tokenize it and return the 
    tokenized string.
    """
    #Create the tokenizer
    tokenizer = nltk.tokenize.ToktokTokenizer()
    
    #Use the tokenizer
    string = tokenizer.tokenize(string, return_str = True)
    
    return string

def stem(string):
    """
    This function will take in a string return a stemmed version of the string.
    """
    
    #Create the stemmer
    ps = nltk.porter.PorterStemmer()
    
    #Apply the stemmer to each word in the string and create a list of stemmed words
    stems = [ps.stem(word) for word in string.split()]
    
    #join the list of stemmed words into a string
    string_stemmed = ' '.join(stems)
    
    return string_stemmed


def lemmatize(string):
    """
    This function takes in a string and returns a lemmatized version of the string.
    """
    
    #Create the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    #Use the lemmatizer on each word in the string to create a list of lemmatized words
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    
    #Join the lemmatized words into one string
    string_lemmatized = ' '.join(lemmas)
    
    return string_lemmatized


def remove_stopwords(string, extra_words = [], exclude_words = []):
    """
    This function will take in a string, filter out stop words from the nltk standard english list 
    as well as any other extra words, and return a version of the text without these stopwords.
    It includes optional paramaters allowing the user to add extra words to remove 
    or to exclude words from the stopword list.
    """
    #Get the standard english stop word list from nltk
    stop_words = stopwords.words('english')
    
    #Add extra words to be removed to the stop word list
    for word in extra_words:
        stop_words.append(word)
    
    #Remove words to be excluded from the stop word list
    for word in exclude_words:
        stop_words.remove(word)
    
    #Create a list of words to be checked by splitting the string
    words = string.split()
    
    #Filter out all of the stop words
    filtered_words = [word for word in words if word not in stop_words]
    
    #Join the list of filtered words into a string
    filtered_string = ' '.join(filtered_words)
    
    return filtered_string



## This is Madeleine's function from exercise review

def prep_article_data(df, column, extra_words=[], exclude_words=[]):
    '''
    This function take in a df and the string name for a text column with 
    option to pass lists for extra_words and exclude_words and
    returns a df with the text article title, original text, stemmed text,
    lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''
    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords,
                                  extra_words=extra_words,
                                  exclude_words=exclude_words)
    
    df['stemmed'] = df['clean'].apply(stem)
    
    df['lemmatized'] = df['clean'].apply(lemmatize)
    
    return df[['title', column,'clean', 'stemmed', 'lemmatized']]


def prep_text(df, column, extra_words=[], exclude_words=[]):
    '''
    This function take in a df and the string name for a text column with 
    option to pass lists for extra_words and exclude_words and
    returns a df with the text article title, original text, stemmed text,
    lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''
    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords,
                                  extra_words=extra_words,
                                  exclude_words=exclude_words)
    
    df['stemmed'] = df['clean'].apply(stem)
    
    df['lemmatized'] = df['clean'].apply(lemmatize)
    
    return df[['label', column,'clean', 'stemmed', 'lemmatized']]