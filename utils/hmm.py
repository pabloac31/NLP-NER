import numpy as np


def evaluate_hmm(Y, Y_hat, ignore_O=True):
    correct = 0
    total = 0
    if ignore_O:
        for y,y_hat in zip(Y,Y_hat):
            for y_k, y_hat_k in zip(y,y_hat):
                if y_k != "O":
                    total +=1
                    if y_hat_k == y_k:
                        correct +=1
        print("Accuracy of posterior decode ignoring 'O' tags:", correct/total)
    else:
        for y,y_hat in zip(Y,Y_hat):
            for y_k, y_hat_k in zip(y,y_hat):
                total +=1
                if y_hat_k == y_k:
                    correct +=1        
        print("Accuracy of posterior decode counting 'O' tags:", correct/total)


def logzero():
    return -np.inf


def safe_log(x):
    print(x)
    if x == 0:
        return logzero()
    return np.log(x)


def logsum_pair(logx, logy):
    """
    Return log(x+y), avoiding arithmetic underflow/overflow.

    logx: log(x)
    logy: log(y)

    Rationale:

    x + y    = e^logx + e^logy
             = e^logx (1 + e^(logy-logx))
    log(x+y) = logx + log(1 + e^(logy-logx)) (1)

    Likewise,
    log(x+y) = logy + log(1 + e^(logx-logy)) (2)

    The computation of the exponential overflows earlier and is less precise
    for big values than for small values. Due to the presence of logy-logx
    (resp. logx-logy), (1) is preferred when logx > logy and (2) is preferred
    otherwise.
    """
    if logx == logzero():
        return logy
    elif logx > logy:
        return logx + np.log1p(np.exp(logy-logx))
    else:
        return logy + np.log1p(np.exp(logx-logy))


def logsum(logv):
    """
    Return log(v[0]+v[1]+...), avoiding arithmetic underflow/overflow.
    """
    res = logzero()
    for val in logv:
        res = logsum_pair(res, val)
    return res


class HMM(object):
    
    def __init__(self, word_to_pos={}, state_to_pos={}):
        self.fitted = False
        self.counts = {"emission": None, "transition":None, "final":None, "initial":None}
        self.probs  = {"emission": None, "transition":None, "final":None, "initial":None}
        self.scores = {"emission": None, "transition":None, "final":None, "initial":None}
        self.decode = set(["posterior", "viterbi"])
        self.word_to_pos  = word_to_pos
        self.state_to_pos = state_to_pos
        self.pos_to_word  = {v: k for k, v in word_to_pos.items()}
        self.pos_to_state = {v: k for k, v in state_to_pos.items()}
    
        self.n_states     = len(state_to_pos)
        self.n_words      = len(word_to_pos)
        self.fitted = False

    def fit(self, observation_lables: list, state_labels: list):
        """
        Computes and saves: counts, probs, scores.
        """
        if self.state_to_pos == None or self.word_to_pos == None:
            print("Error state_to_pos or word_to_pos needed to be defined")
            return
            
        self.counts = self.sufficient_statistics_hmm(observation_lables, state_labels)       
        self.probs  = self.compute_probs(self.counts)  
        self.scores = self.compute_scores(self.probs)  
        self.fitted = True
        
    def sufficient_statistics_hmm(self, observation_lables, state_labels):

        state_to_pos, word_to_pos = self.state_to_pos, self.word_to_pos
        
        # custom update functions (adapted to the HMM class nomenclature)
        def update_initial_counts(initial_counts, seq_y, state_to_pos):
            pos = state_to_pos[seq_y[0]]
            initial_counts[pos] += 1

        def update_transition_counts(transition_counts, seq_y, state_to_pos):
            L = len(seq_y)
            for i in range(L-1):
                # he intercambiado pos_i y pos_j pa que dé lo que a él
                pos_j = state_to_pos[seq_y[i]]
                pos_i = state_to_pos[seq_y[i+1]]
                transition_counts[pos_i, pos_j] += 1

        def update_emission_counts(emission_counts, seq_x, seq_y, state_to_pos, word_to_pos):
            L = len(seq_x)
            for i in range(L):
                pos_i = state_to_pos[seq_y[i]]
                pos_j = word_to_pos[seq_x[i]]
                emission_counts[pos_i, pos_j] += 1

        def update_final_counts(final_counts, seq_y, state_to_pos):
            pos = state_to_pos[seq_y[-1]]
            final_counts[pos] += 1    

        n_states = len(state_to_pos)
        n_words  = len(word_to_pos)
        initial_counts      = np.zeros((n_states))
        transition_counts   = np.zeros((n_states, n_states))
        final_counts        = np.zeros((n_states))
        emission_counts     = np.zeros((n_states, n_words))

        for seq_x, seq_y in zip(observation_lables, state_labels):
            update_initial_counts(initial_counts, seq_y, state_to_pos)
            update_transition_counts(transition_counts, seq_y,  state_to_pos)
            update_emission_counts(emission_counts, seq_x, seq_y, state_to_pos, word_to_pos) 
            update_final_counts(final_counts, seq_y,  state_to_pos) 

        return {"emission":   emission_counts, 
                "transition": transition_counts,
                "final":      final_counts, 
                "initial":    initial_counts}
    
    def compute_probs(self, counts):
        
        initial_counts    = counts['initial']
        transition_counts = counts['transition']
        emission_counts   = counts['emission']
        final_counts      = counts['final']

        initial_probs    = (initial_counts / np.sum(initial_counts))
        transition_probs = transition_counts/(np.sum(transition_counts,0) + final_counts)
        final_probs      = final_counts/(np.sum(transition_counts, 0) + final_counts )
        emission_probs   = (emission_counts.T / np.sum(emission_counts, 1)).T
    
        return {"emission":   emission_probs, 
                "transition": transition_probs,
                "final":      final_probs, 
                "initial":    initial_probs}
    
    def compute_scores(self, probs):
         return {"emission":   np.log(probs["emission"]), 
                 "transition": np.log(probs["transition"]),
                 "final":      np.log(probs["final"]), 
                 "initial":    np.log(probs["initial"])}
        
    def forward_computations(self, x: list):
        forward_x = None
        return forward_x
    
    def backward_computations(self, x:list):
        backward_x = None
        return backward_x
    
    def log_forward_computations(self, x: list):
        """
        Compute the log_forward computations

        Assume there are S possible states and a sequence of length N.
        This method will compute iteritavely the log_forward quantities.

        * log_f is a S x N Array.
        * log_f_x[:,i] will contain the forward quantities at position i.
        * log_f_x[:,i] is a vector of size S.
        
        Returns
        - log_f_x: Array of size S x N
        """ 
        n_x = len(x)
        
        # log_f_x initialized to -Inf because log(0) = -Inf
        log_f_x = np.zeros((self.n_states, n_x)) - np.Inf
        x_emission_scores = np.array([self.scores['emission'][:, self.word_to_pos[w]] for w in x]).T
        
        log_f_x[:,0] = x_emission_scores[:, 0] + self.scores['initial']
        for n in range(1, n_x):
            for s in range(self.n_states):
                # log_Pt indexed like below or I get sth different from him
                log_Pt = self.scores['transition'][s,:]
                log_fm1 = log_f_x[:, n-1]
                logsum_arg = log_Pt + log_fm1
                log_f_x[s, n] = x_emission_scores[s, n] + logsum(logsum_arg)
            
        log_likelihood = logsum(self.scores["final"]+log_f_x[:, -1])
        #self.log_f_x = log_f_x
        
        return log_f_x, log_likelihood
    
    
    def log_backward_computations(self, x: list):
        n_x = len(x)
        
        # log_b_x initialized to -Inf because log(0) = -Inf
        log_b_x = np.zeros((self.n_states, n_x)) - np.Inf
        x_emission_scores = np.array([self.scores['emission'][:, self.word_to_pos[w]] for w in x]).T
        log_b_x[:,-1] = self.scores['final']

        for n in range(n_x-2, -1, -1):
            for s in range(self.n_states):
                # log_Pt indexed like below or I get sth different from him
                log_Pt = self.scores['transition'][:, s]
                log_bp1 = log_b_x[:, n+1]
                logsum_arg = log_Pt + log_bp1 + x_emission_scores[:, n+1]
                log_b_x[s, n] =  logsum(logsum_arg)
        
        logsum_arg = self.scores['initial'] + log_b_x[:, 0] + x_emission_scores[:, 0]
        log_likelihood = logsum(logsum_arg)
        #self.log_b_x = log_b_x   
        return log_b_x, log_likelihood
        
    def predict_labels(self, x: list, decode="posterior"):
        """
        Retuns a sequence of states for each word in **x**.
        The output depends on the **decode** method chosen.
        """
        assert decode in self.decode, "decode `{}` is not valid".format(decode)
        
        if decode == 'posterior':
            return self.posterior_decode(x)
        
        if decode == 'viterbi':
            return self.viterbi_decode(x)

    def compute_state_posteriors(self, x:list):
        log_f_x, log_likelihood = self.log_forward_computations(x)
        log_b_x, log_likelihood = self.log_backward_computations(x)
        state_posteriors = np.zeros((self.n_states, len(x)))
        
        for pos in range(len(x)):
            state_posteriors[:, pos] = log_f_x[:, pos] + log_b_x[:, pos] - log_likelihood
        return state_posteriors

    def posterior_decode(self, x: list, decode_states=True):
        log_f_x, log_likelihood = self.log_forward_computations(x)
        log_b_x, log_likelihood = self.log_backward_computations(x)
        idxs = np.argmax(log_f_x + log_b_x, axis=0) #for max along columns 
        y_hat = [self.pos_to_state[i] for i in idxs]
        return y_hat