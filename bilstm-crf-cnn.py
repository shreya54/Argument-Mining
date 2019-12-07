import os
import sys
import src.char_cnn as charCNN
import src.char_embedd as charEMBEDD
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"

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

def create_emb_layer(weights_matrix, non_trainable=True):
    print "create_emb_layer"
    weights_matrix=torch.Tensor(weights_matrix)
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        print "embedding layer is non trainable"
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
def copylists(list_):
    ans = []
    for i in range(1, len(list_)):
        ans.append(list_[i])
    return ans


def get_training_data(trainFILE):
    sentences = []
    sent = []
    sentences_idx = []
    idx = []
    word_toidx = {}
    label_toidx = {}
    new_para = True
    count_word = 0
    count_label = 0
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
                label = line[4].lower()
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
                        sent.append(word_toidx["<STOP>"])
                        idx.append(label_toidx["<STOP>"])
                        sentences.append(sent)
                        sentences_idx.append(idx)
                    sent = [word_toidx["<START>"]]
                    idx  = [label_toidx["<START>"]]
                    sent.append(word)
                    idx.append(label)
                    new_para = False
                else:
                    sent.append(word)
                    idx.append(label)
    for item in sentences_idx:
        if len(item) <= 3:
            print item
    print len(sentences)
    word_len = 15
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

    print "max length", max_len
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
    return word_toidx,label_toidx,sentences_idx,sentences, word_len




class BiLSTM_CRF_CNN(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,weights_matrix, char_embeddings_dim, word_len):
        super(BiLSTM_CRF_CNN, self).__init__()
        #self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        #self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.char_embeddings_layer = CharEMBEDD(gpu, char_embeddings_dim, False,
                                                 word_len, word_seq_indexer.get_unique_characters_list())

        self.char_cnn_layer = CharCNN(gpu, char_embeddings_dim, char_cnn_filter_num, char_window_size,
                                           word_len)
        self.word_embeds, vocab_size, embedding_dim = create_emb_layer(weights_matrix, True)
        #self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        #print "word_embeds.shape() " , self.word_embeds.shape
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        #print self.transitions.data
        print "self.transition.shape", self.transitions.shape

        self.hidden = self.init_hidden()
        print torch.randn(2, 1, self.hidden_dim // 2).shape

        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        self.softmax = nn.Softmax(dim=2)

        #print self.hidden

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))


    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        z_embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        z_char_embed = self.char_embeddings_layer(word_sequences)
        z_char_embed_d = self.dropout(z_char_embed)
        z_char_cnn = self.char_cnn_layer(z_char_embed_d)
        z = torch.cat((z_embeds, z_char_cnn), dim=2)
        lstm_out, self.hidden = self.lstm(z, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


if __name__=="__main__":
    trainFILE = sys.argv[1]
    testFILE = sys.argv[2]
    embedding_path = sys.argv[3]
    word_dict, label_dict , label_idxs, sentence_idxs, word_len= get_training_data(trainFILE)
    test_sent, test_labels = gettestdata(word_dict, label_dict, testFILE)

    embedd_dict, embedd_dim=load_word_embedding_dict(embedding_path,embedd_dim=100)

    EMBEDDING_DIM = 100
    HIDDEN_DIM = 4
    # len(label_dict) 71
    # len(word_dict) 7301

    matrix_len = len(word_dict)
    weights_matrix = np.zeros((matrix_len, 100))
    words_found = 0

    inv_word_dict = {}
    for word in word_dict:
        lab = word_dict[word]
        inv_word_dict[lab] = word

    for i in range(0, len(inv_word_dict)):
        try:
            weights_matrix[i] = embedd_dict[inv_word_dict[i]]
        except:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM, ))
    
    print len(label_dict)


    model = BiLSTM_CRF_CNN(len(word_dict), label_dict, EMBEDDING_DIM, HIDDEN_DIM, weights_matrix, 25, word_len)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        print sentence_idxs[1]
        print label_idxs[1]
        precheck_sent=torch.tensor(sentence_idxs[1],dtype=torch.long)

        print "model(precheck_sent): ", model(precheck_sent)



    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(0,10):  # again, normally you would NOT do 300 epochs, it is toy data
        print "epoch: " , epoch 
        for i in range(0,len(sentence_idxs)):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in=torch.tensor(sentence_idxs[i],dtype=torch.long)
            targets=torch.tensor(label_idxs[i],dtype=torch.long)
            #print "sentence_in" , sentence_in
            #targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            #print "targets " , targets
            #print "before loss"
            #print "i === ", i
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)
            print "loss shape - ", loss
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            
            l = optimizer.step()
            print l

    # Check predictions after training
    with torch.no_grad():
        #load test data

        print sentence_idxs[1]
        print label_idxs[1]
        precheck_sent=torch.tensor(sentence_idxs[1],dtype=torch.long)

        print "model(precheck_sent): ", model(precheck_sent)





