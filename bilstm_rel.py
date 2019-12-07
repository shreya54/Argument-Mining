from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
import sys
import numpy as np


def load_word_embedding_dict(embedding_path,embedd_dim=100):
    print "loading dict"
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue

            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim], dtype='float32')
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict,embedd_dim


def copylists(list_):
    ans = []
    for i in range(1, len(list_)):
        ans.append(list_[i])
    return ans

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)



def main(trainFILE):
    sentences = []
    sent = []
    sentences_idx = []
    idx = []
    word_toidx = {}
    label_toidx = {}
    new_para = True
    count_word = 3
    count_label = 3
    START_TAG = "<START>" #--------1
    STOP_TAG = "<STOP>"   #--------2
    word_toidx["unk"] = 0
    label_toidx["unk"] = 0
    word_toidx["<START>"] = 1
    word_toidx["<STOP>"] = 2
    label_toidx["<START>"] = 1
    label_toidx["<STOP>"] = 2

    #tag_to_idx = {"B": 3, "I": 4, "O": 5, START_TAG: 1, STOP_TAG: 2,"unk" : 0}
    count_space = 0
    with open(trainFILE) as fp:
        for line in fp:
            line.decode('utf-8')
            line = line.strip()
            line = line.split()
            if len(line)==0:
                new_para = True
                continue
            else:
                word = line[1].lower()
                label = line[4]
                if word in word_toidx:
                    word = word_toidx[word]
                else:
                    word_toidx[word] = count_word
                    count_word = count_word + 1
                    word = word_toidx[word]

                if label in label_toidx:
                    label = label_toidx[label]
                else:
                    label_toidx[label] = count_label
                    count_label = count_label + 1
                    label = label_toidx[label]


                if new_para==True:
                    if len(sent)!=0:
                        sentences.append(sent)
                        sentences_idx.append(idx)
                    sent=[]
                    idx=[]
                    sent.append(word)
                    idx.append(label)
                    new_para = False
                else:
                    sent.append(word)
                    idx.append(label)
    print len(sentences)
    #Finding maximum length of a para
    max_len = 0
    empty_in = []
    for i in range(0,len(sentences)):
        #print "i=", i, "  ", len(sentences[i])
        if len(sentences[i]) == 0:
            empty_in.append(i)
        else:
            if(len(sentences[i])) > max_len:
                max_len = len(sentences[i])

    print "max length train", max_len
    #padding
    for i in range(1, len(sentences)):
        while(len(sentences[i])<max_len):
            sentences[i].append(0)
            sentences_idx[i].append(sentences_idx[i][-1])
    sentences = copylists(sentences)
    sentences_idx = copylists(sentences_idx)
    #Padding of the paras
    print "length", len(sentences)
    #[1, 567, 13, 6847, 817, 122, 56, 57, 25, 846, 30, 131, 13, 3987, 16, 858, 401, 1238, 3092, 13, 70, 133, 7, 346, 78, 30, 65, 138, 131, 1377, 54, 7, 3244, 1581, 21, 1106, 23, 711, 4, 753, 70, 445, 117, 6848, 25, 140, 521, 122, 62, 13, 252, 74, 333, 407, 1167, 122, 686, 2894, 1707, 25, 270, 30, 852, 6849, 68, 54, 5, 398, 122, 21, 2027, 23, 6850, 1158, 58, 81, 25, 2]
    return word_toidx,label_toidx,sentences_idx,sentences,max_len

def main_test(testFILE,word_toidx,tag_to_idx,max_len):
    sentences = []
    sent = []
    sentences_idx = []
    idx = []
    #word_toidx = {}
    #tag_to_idx = {}
    new_para = True
    count_word = 3
    count_label = 3
    START_TAG = "<START>" #--------1
    STOP_TAG = "<STOP>"   #--------2
    word_toidx["unk"] = 0
    #tag_to_idx["unk"] = 0
    word_toidx["<START>"] = 1
    word_toidx["<STOP>"] = 2
    count_space = 0
    with open(testFILE) as fp:
        for line in fp:
            line.decode('utf-8')
            line = line.strip()
            line = line.split()
            if len(line)==0:
                new_para = True
                continue
            else:
                word = line[1].lower()
                label = line[4].split('-')
                if word in word_toidx:
                    word = word_toidx[word]
                else:
                    word = word_toidx["unk"]

                label = tag_to_idx[label[0]]

                if new_para==True:
                    if len(sent)!=0:
                        sentences.append(sent)
                        sentences_idx.append(idx)
                    sent=[]
                    idx=[]
                    sent.append(word)
                    idx.append(label)
                    new_para = False
                else:
                    sent.append(word)
                    idx.append(label)
    print len(sentences)
    #padding
    for i in range(1, len(sentences)):
        while(len(sentences[i])<max_len):
            sentences[i].append(0)
            sentences_idx[i].append(sentences_idx[i][-1])
    sentences = copylists(sentences)
    sentences_idx = copylists(sentences_idx)
    #Padding of the paras
    #[1, 567, 13, 6847, 817, 122, 56, 57, 25, 846, 30, 131, 13, 3987, 16, 858, 401, 1238, 3092, 13, 70, 133, 7, 346, 78, 30, 65, 138, 131, 1377, 54, 7, 3244, 1581, 21, 1106, 23, 711, 4, 753, 70, 445, 117, 6848, 25, 140, 521, 122, 62, 13, 252, 74, 333, 407, 1167, 122, 686, 2894, 1707, 25, 270, 30, 852, 6849, 68, 54, 5, 398, 122, 21, 2027, 23, 6850, 1158, 58, 81, 25, 2]
    return sentences_idx,sentences

def make_wt_matrix(word_dict,embedd_dict,embedd_dim):
    matrix_len = len(word_dict)
    weights_matrix = np.zeros((matrix_len, 100))
    #words_found = 0
    inv_word_dict = {}
    for word in word_dict:
        lab = word_dict[word]
        inv_word_dict[lab] = word

    for i in range(0, len(inv_word_dict)):
        try:
            weights_matrix[i] = embedd_dict[inv_word_dict[i]]
        except:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(embedd_dim, ))

    return weights_matrix


if __name__=="__main__":
    trainFILE = sys.argv[1]
    testFILE = sys.argv[3]
    embedding_path = sys.argv[2]
    word_dict, label_dict , label_idxs,sentence_idxs,max_len,X_lengths= main(trainFILE)
    label_idxs_test,sentence_idxs_test = main_test(testFILE,word_dict,label_dict,max_len)

    embedd_dict, embedd_dim=load_word_embedding_dict(embedding_path,embedd_dim=100)
    weights_matrix=make_wt_matrix(word_dict,embedd_dict,embedd_dim)


    e = Embedding(len(word_dict), 100, weights=[weights_matrix], input_length=max_len, trainable=False)

    model = Sequential()
    model.add(InputLayer(input_shape=(max_len, )))
    model.add(e)
    #model.add(Embedding(len(word_dict), 100))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(label_dict))))
    model.add(Activation('softmax'))
 
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy'])
 
    model.summary()

    print "label_idxs: " , label_idxs[0]
    label_idxs=np.array(label_idxs)
    sentence_idxs=np.array(sentence_idxs)
    label_idxs_test=np.array(label_idxs_test)
    sentence_idxs_test=np.array(sentence_idxs_test)

    cat_train_tags_y = to_categorical(label_idxs, len(label_dict))
    print(cat_train_tags_y[0])
    model.fit(sentence_idxs, to_categorical(label_idxs, len(label_dict)), batch_size=128, epochs=1, validation_split=0.2)
    scores = model.evaluate(sentence_idxs, to_categorical(label_idxs, len(label_dict)))

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))