import nltk
import sys
import os
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
import math
import re


FILE_MATCHES = 1
SENTENCE_MATCHES = 1

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    
    # for filename in os.listdir(directory):
    #     with open(os.path.join(directory, filename)) as f:

    #         print(f"Hey! {f.name}")
    for _, _, file_list in os.walk(directory): # name it file_list to avoid conflicts with files
        for filename in file_list:
            with open(f"corpus/{filename}") as f:
                lines = f.readlines() # get the sentences in the file
                files[filename] = str(lines)
    return files
         

    
def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    cleaned_word_list = []
    stops = set(stopwords.words("english"))
    # for word in document: # this loop caused a bug which a single word becomes separated by each letter
    document = str(document)
    document_no_punc = re.sub(r'[^\w\s]', '', document)
    token_words = nltk.word_tokenize(document_no_punc.lower())
    for w in token_words:
           if not w in string.punctuation and not w in stops:
              cleaned_word_list.append(w.lower())
    
    return cleaned_word_list
    
    
def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    
    # calculate idfs
    idfs = dict()
    for file_names, words in documents.items():
        for word in words:
            f = sum(word in documents[filename] for filename in documents )
            idf = math.log(len(documents) / f)
            idfs[word] = idf
    
    return idfs
    
    
def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # top_list = []
    # tfidfs = dict()
    # frequencies = dict()
    # # count frequencies
    # for file_names, words in files.items():
    #     for word in words:
    #         if word not in frequencies:
    #             frequencies[word] = 1
    #         else:
    #             frequencies[word] += 1
    #         for word, tf in frequencies.items():
    #             tfidfs[].append((word, tf * idfs[word]))
    # print(tfidfs)
    
    def check_priority(file):
        priority = 0
        for word in query:
            tf = sum( w == word for w in files[file])
            priority += tf * idfs[word]
        return priority
    
    sorted_files = sorted(files.keys(), key=lambda file: check_priority(file), reverse=True)
    return sorted_files[:n]


    

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    def check_priority(sentence):
        # priority = 0
        # count = 0
        # for word in query:
        #     if word in sentence:
        #         count += 1
        #         density = count / len(sentence)
        file = sentences[sentence]
        density = 0
        idf_total = sum(idfs[word] for word in set(query) if word in set(file))
        file_len = len(file)

        for word in file:
            if word in query:
                density += 1
                density = density / file_len
        return idf_total, density
    
    sortd_sentences = sorted(sentences.keys(), key=lambda s: check_priority(s), reverse=True)[:n]
    return sortd_sentences


if __name__ == "__main__":
    main()
