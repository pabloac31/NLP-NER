import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from utils.skseq.sequences.sequence import Sequence


def evaluate_corpus(sequences, sequences_predictions):
    """Evaluate classification accuracy at corpus level, comparing with
    gold standard."""
    total = 0.0
    correct = 0.0
    for i, sequence in enumerate(sequences):
        pred = sequences_predictions[i]
        for j, y_hat in enumerate(pred.y):
            if sequence.y[j] != 0: #0 is the index of the "O" tag
                if sequence.y[j] == y_hat:
                    correct += 1
                total += 1
    return correct / total


def show_confusion_matrix(sequences, preds, sp=None, hmm=False, normalize=False, positions=None, labels=None):
    if hmm:
        y_true = [item for sublist in sequences for item in sublist]
        y_pred = [item for sublist in preds for item in sublist]
    else:
        y_true = []
        y_pred = []
        for seq, pred in zip(sequences, preds):
            y_true.extend(seq.y)
            y_pred.extend(pred.y.tolist())

    cm = confusion_matrix(y_true, y_pred)

    threshold = 24953
    cm_clipped = np.clip(cm, a_min=0, a_max=threshold)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm_clipped, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title("Confusion matrix")
    #plt.colorbar()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = threshold / 1.5 if normalize else threshold / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    if (positions==None) | (labels==None):
        positions = list(sp.state_labels.values())
        labels = list(sp.state_labels.keys())

    plt.xticks(positions, labels)
    plt.yticks(positions, labels)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.show()


def get_f1_score(sequences, preds, hmm=False):
    if hmm:
        y_true = [item for sublist in sequences for item in sublist]
        y_pred = [item for sublist in preds for item in sublist]
    else:
        y_true = []
        y_pred = []
        for seq, pred in zip(sequences, preds):
            y_true.extend(seq.y)
            y_pred.extend(pred.y.tolist())

    return f1_score(y_true, y_pred, average='weighted')


def tiny_test(model, train_seq=None, hmm=False, state_to_pos=None):

    sentences = [
        "The programmers from Barcelona might write a sentence without a spell checker.",
        "The programmers from Barchelona cannot write a sentence without a spell checker.",
        "Jack London went to Parris.",
        "Jack London went to Paris.",
        "Bill gates and Steve jobs never though Microsoft would become such a big company.",
        "Bill Gates and Steve Jobs never though Microsof would become such a big company.",
        "The president of U.S.A though they could win the war.",
        "The president of the United States of America though they could win the war.",
        "The king of Saudi Arabia wanted total control.",
        "Robin does not want to go to Saudi Arabia.",
        "Apple is a great company.",
        "I really love apples and oranges.",
        "Alice and Henry went to the Microsoft store to buy a new computer during their trip to New York."
    ]
    
    y_pred = []
    if hmm:
        for p in sentences:
            pred = model.predict_labels(p.split())
            seq = Sequence(x=p.split(), y=pred)
            print(seq, '\n') 
            y_pred.extend(pred)
        y_pred = [state_to_pos[w] for w in y_pred]
    else:
        preds = []
        for p in sentences:
            seq = Sequence(x=p.split(), y=[int(0) for w in p.split()])
            pred = model.viterbi_decode(seq)[0]
            preds.append(pred)
            y_pred.extend(pred.y.tolist())
            print(pred.to_words(train_seq, only_tag_translation=True), '\n')

    # evaluate results
    y_true = [0,0,0,1,0,0,0,0,0,0,0,0] + [0,0,0,0,0,0,0,0,0,0,0,0]
    y_true += [6,7,0,0,0] + [6,7,0,0,1]
    y_true += [6,7,0,6,7,0,0,4,0,0,0,0,0,0] + [6,7,0,6,7,0,0,0,0,0,0,0,0,0]
    y_true += [0,0,0,1,0,0,0,0,0,0] + [0,0,0,0,1,5,5,5,0,0,0,0,0,0]
    y_true += [0,0,0,1,5,0,0,0] + [6,0,0,0,0,0,0,1,5]
    y_true += [4,0,0,0,0] + [0,0,0,0,0,0]
    y_true += [6,0,6,0,0,0,4,0,0,0,0,0,0,0,0,0,0,1,5]

    correct = total = 0
    for y, y_hat in zip(y_true, y_pred):
        if y != 0: #0 is the index of the "O" tag
            if y == y_hat:
                correct += 1
            total += 1
    print("\n===============================")
    print(f"Accuracy in TINY TEST = {round(correct/total, 4)}")
    print("===============================")



class BiLSTM_CRF_v2(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF_v2, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
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


        self.hidden = self.init_hidden()

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
        forward_var = forward_var

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


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
