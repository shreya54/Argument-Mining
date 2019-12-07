import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np

from sklearn.metrics import f1_score


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


def main(trainFILE):
    sentences = []
    sent = []
    sentences_idx = []
    idx = []
    word_toidx = {}
    tag_to_idx = {}
    new_para = True
    count_word = 3
    count_label = 3
    START_TAG = "<START>" #--------1
    STOP_TAG = "<STOP>"   #--------2
    word_toidx["unk"] = 0
    #tag_to_idx["unk"] = 0
    word_toidx["<START>"] = 1
    word_toidx["<STOP>"] = 2
    #label_toidx["<START>"] = 1
    #label_toidx["<STOP>"] = 2

    tag_to_idx = {"B": 3, "I": 4, "O": 5, START_TAG: 1, STOP_TAG: 2,"unk" : 0}
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
                label = line[4].split('-')
                if word in word_toidx:
                    word = word_toidx[word]
                else:
                    word_toidx[word] = count_word
                    count_word = count_word + 1
                    word = word_toidx[word]

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
    return word_toidx,tag_to_idx,sentences_idx,sentences,max_len

def main_test(testFILE,embedd_dict,embedd_dim,word_toidx,tag_to_idx,max_len):
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
    #Finding maximum length of a para
    # max_len = 0
    # empty_in = []
    # for i in range(0,len(sentences)):
    #     #print "i=", i, "  ", len(sentences[i])
    #     if len(sentences[i]) == 0:
    #         empty_in.append(i)
    #     else:
    #         if(len(sentences[i])) > max_len:
    #             max_len = len(sentences[i])

    print "max length test", max_len
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
    return sentences_idx,sentences



class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,weights_matrix):
        super(BiLSTM_CRF, self).__init__()
        #self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        #self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix

        self.tagset_size = len(tag_to_ix)

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
        #print "self.transition.shape", self.transitions.shape

        self.hidden = self.init_hidden()
        #print torch.randn(2, 1, self.hidden_dim // 2).shape
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
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
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

def accuracy(out, labels,s,label_size):
  #outputs = np.argmax(out)
  #print "np.argmax(out): outputs" , outputs
  out=np.array(out)
  labels=np.array(labels)

  s+=(out==labels).sum()
  #print "outputs.size: " , len(out) , "label_size: " , len(labels) 
  label_size+=float(len(labels))
  #print "s:", s,"label_size", label_size
  return s,label_size

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
            weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM, ))

    return weights_matrix


if __name__=="__main__":
    trainFILE = sys.argv[1]
    testFILE = sys.argv[3]
    embedding_path = sys.argv[2]
    word_dict, label_dict , label_idxs,sentence_idxs,max_len= main(trainFILE)
    
    #print len(word_dict)
    k = np.zeros(len(word_dict) + 1)
    for item in word_dict:
        l = word_dict[item]
        k[l] = 1
    for i in range(0, len(k)):
        if k[i] == 0:
            print "index not all", i
    embedd_dict, embedd_dim=load_word_embedding_dict(embedding_path,embedd_dim=100)

    EMBEDDING_DIM = 100
    HIDDEN_DIM = 4
    # len(label_dict) 71
    # len(word_dict) 7301

    weights_matrix=make_wt_matrix(word_dict,embedd_dict,embedd_dim)



    model = BiLSTM_CRF(len(word_dict), label_dict, EMBEDDING_DIM, HIDDEN_DIM, weights_matrix)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    print "making test data"
    label_idxs_test,sentence_idxs_test = main_test(testFILE,embedd_dict,embedd_dim,word_dict,label_dict,max_len)

    # Check predictions before training
    with torch.no_grad():
        print sentence_idxs[1]
        print label_idxs[1]
        print "checking predictions before training"
        precheck_sent=torch.tensor(sentence_idxs[1],dtype=torch.long)

        output=model(precheck_sent)
        print "model(precheck_sent): " , output


    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(0,50):  # again, normally you would NOT do 300 epochs, it is toy data
        print "epoch: " , epoch 
        for i in range(0,len(sentence_idxs)):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in=torch.tensor(sentence_idxs[i],dtype=torch.long)
            targets=torch.tensor(label_idxs[i],dtype=torch.long)

            loss = model.neg_log_likelihood(sentence_in, targets)
         
            loss.backward()
            optimizer.step()

    # Check predictions after training
    s=0
    label_size=0
    with torch.no_grad():
        for i in range(0,len(sentence_idxs_test)):

            precheck_sent=torch.tensor(sentence_idxs_test[i],dtype=torch.long)

            output_idxs=model(precheck_sent)
           
            s,label_size=accuracy(output_idxs[1],label_idxs_test[i],s,label_size)
            temp_acc=s/label_size
            print "i: " , i , "accuracy: " , temp_acc
            print "label_idxs_test: " , label_idxs_test[i]
            print "output: ", output_idxs[1]
            
        accuracy=s/label_size
        print "final accuracy: " , accuracy





