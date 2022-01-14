from collections import defaultdict
import numpy as np
from hmm import HMM

def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)
    
    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    ind = 0
    for word in unique_words:
        if word not in word2idx:
            word2idx[word] = ind
            ind+=1
    
    for i, tag in enumerate(tags):
        tag2idx[tag] = i
    
    tag_for_word = defaultdict(lambda: defaultdict(int))
    tag_from_to = defaultdict(lambda: defaultdict(int)) # current tag as rows, source tag in cols

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    N = len(train_data)
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################
    start_tags =  defaultdict(int)
    tags_counter = defaultdict(int)
    tags_transition_total = defaultdict(int)


    for sent in train_data:
        prev = ''
        start_tags[sent.tags[0]]+=1
        for i, word in enumerate(sent.words):
            tags_counter[sent.tags[i]]+=1
            tags_transition_total[prev]+=1
            tag_from_to[prev][sent.tags[i]]+=1
            tag_for_word[sent.tags[i]][word]+=1
            prev = sent.tags[i]
    
    for tag in tags:
        pi[tag2idx[tag]] = start_tags[tag] / N
        
    for curr_tag in tags:
        for next_tag in tags:
            A[tag2idx[curr_tag]][tag2idx[next_tag]] = tag_from_to[curr_tag][next_tag] / tags_transition_total[curr_tag]
    
    for curr_tag in tags:
        for word, count in tag_for_word[curr_tag].items():
            B[tag2idx[curr_tag]][word2idx[word]] = count / tags_counter[curr_tag]


    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################
    # Shape of Alpha in hmm.py is S x L
    S = len(tags)

    # Since for each sentence, there are many words i,e obs, we need to find the last index we used to encode the last word
    req_index = 0
    for x in model.obs_dict:
        req_index = max(req_index, model.obs_dict[x])
    ind = req_index + 1
    for sent in test_data:
        for word in sent.words:
            if word in model.obs_dict:
                continue
            else:
                model.obs_dict[word] = ind
                # create dummy Beta_s for each s
                model.B = np.append(model.B, np.full((S, 1), 10**-6), axis=1)
                ind+=1
        tagging.append(model.viterbi(sent.words))            
    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
