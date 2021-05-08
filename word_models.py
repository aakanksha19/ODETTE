import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class EventExtractor(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, output_size, dropout, is_bidir=True, rep_learner='word', num_layers=3, pos_vocab_size=None):
        super(EventExtractor, self).__init__()
        self.rep_type = rep_learner
        if rep_learner == 'word':
            self.rep_learner = BiLSTM(vocab_size, emb_size, hidden_size, dropout, is_bidir)
        if rep_learner == 'delex':
            self.rep_learner = DelexBiLSTM(vocab_size, emb_size, hidden_size, dropout, is_bidir)
        if rep_learner == 'bert-bilstm':
            self.rep_learner = BiLSTM(vocab_size, emb_size, hidden_size, dropout, is_bidir, use_bert_embs=True)
        if rep_learner == 'bert-mlp':
            self.rep_learner = MLP(vocab_size, emb_size, hidden_size, num_layers, dropout, use_bert_embs=True)
        if rep_learner == 'pos':
            self.rep_learner = PosBiLSTM(vocab_size, pos_vocab_size, emb_size, hidden_size, dropout, is_bidir)
        if is_bidir:
            self.rep_size = 2 * hidden_size
        else:
            self.rep_size = hidden_size
        self.classifier = EventClassifier(self.rep_size, output_size)

    def forward(self, sents, lengths, masks, pos=None):
        rep_outputs = []
        if self.rep_type == 'word' or self.rep_type == 'delex' or self.rep_type.startswith('bert'):
            rep_outputs = self.rep_learner(sents, lengths, masks)
        elif self.rep_type == 'pos':
            rep_outputs = self.rep_learner(sents, pos, lengths, masks)
        self.reps = rep_outputs
        if 'mlp' not in self.rep_type and self.rep_learner.is_bidir:
            rep_outputs = rep_outputs.contiguous().view(-1, 2 * self.rep_learner.hidden_size)
        else:
            rep_outputs = rep_outputs.contiguous().view(-1, self.rep_learner.hidden_size)
        rep_outputs = rep_outputs * masks.contiguous().view(-1, 1)
        class_outputs = self.classifier(rep_outputs)
        return class_outputs  # Remove rep_outputs after dumping BiLSTM outputs

class EventClassifier(nn.Module):

    def __init__(self, input_size, output_size):
        super(EventClassifier, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, features):
        return self.linear(features)

class RepresentationLearner(nn.Module):

    def __init__(self, vocab_size, emb_size):
        super(RepresentationLearner, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.embeddings = nn.Embedding(vocab_size, emb_size, padding_idx=0)

    def load_embeddings(self, file_path, vocab):
        reader = open(file_path, 'r')
        embs = np.random.normal(size=(self.vocab_size, self.emb_size))
        embs[0, :] = np.zeros((1, self.emb_size))
        for line in reader:
            data = line.strip().split()
            word = data[0]
            if word not in vocab:
                continue
            try:
                embs[vocab[word], :] = np.array(data[1:])
            except Exception as e:
                print(e)
                continue
        self.embeddings.load_state_dict({'weight': torch.FloatTensor(embs)})
        self.embeddings.weight.requires_grad = False

class MLP(RepresentationLearner):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, dropout, use_bert_embs=False):
        super(MLP, self).__init__(vocab_size, emb_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bert_embs = use_bert_embs

        self.layers = nn.ModuleList([nn.Linear(emb_size, hidden_size)])
        self.layers.append(nn.Tanh())
        for i in range(num_layers-1):
            self.layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh()])

    def forward(self, sents, lengths, masks):
        seq_len = sents.size()[1]
        features = sents.contiguous().view(-1, self.emb_size)
        for i in range(len(self.layers)):
            features = self.layers[i](features)
        features = features.view(-1, seq_len, self.hidden_size)
        return features

class BiLSTM(RepresentationLearner):

    def __init__(self, vocab_size, emb_size, hidden_size, dropout, is_bidir, use_bert_embs=False):
        super(BiLSTM, self).__init__(vocab_size, emb_size)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.is_bidir = is_bidir
        self.use_bert_embs = use_bert_embs

        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=1, batch_first=True, bidirectional=is_bidir)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size):
        hidden_a = torch.randn(2, batch_size, self.hidden_size)
        hidden_b = torch.randn(2, batch_size, self.hidden_size)

        if not self.is_bidir:
            hidden_a = torch.randn(1, batch_size, self.hidden_size)
            hidden_b = torch.randn(1, batch_size, self.hidden_size)

        if torch.cuda.is_available():
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, sents, lengths, masks):
        embedded_sents = sents
        if not self.use_bert_embs:
            embedded_sents = self.embeddings(sents)
        embedded_sents = self.dropout(embedded_sents)

        self.hidden = self.init_hidden(sents.size()[0])
        packed_sents = nn.utils.rnn.pack_padded_sequence(embedded_sents, lengths, batch_first=True)
        lstm_outputs, self.hidden = self.lstm(packed_sents, self.hidden)
        lstm_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_outputs, batch_first=True)
        return lstm_outputs

class DelexBiLSTM(RepresentationLearner):

    def __init__(self, vocab_size, emb_size, hidden_size, dropout, is_bidir):
        super(DelexBiLSTM, self).__init__(vocab_size, emb_size)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.is_bidir = is_bidir
        self.embeddings.weight.requires_grad = True
        
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=1, batch_first=True, bidirectional=is_bidir)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size):
        hidden_a = torch.randn(2, batch_size, self.hidden_size)
        hidden_b = torch.randn(2, batch_size, self.hidden_size)

        if not self.is_bidir:
            hidden_a = torch.randn(1, batch_size, self.hidden_size)
            hidden_b = torch.randn(1, batch_size, self.hidden_size)

        if torch.cuda.is_available():
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, sents, lengths, masks):
        embedded_sents = self.embeddings(sents)
        embedded_sents = self.dropout(embedded_sents)

        self.hidden = self.init_hidden(sents.size()[0])
        packed_sents = nn.utils.rnn.pack_padded_sequence(embedded_sents, lengths, batch_first=True)
        lstm_outputs, self.hidden = self.lstm(packed_sents, self.hidden)
        lstm_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_outputs, batch_first=True)

        return lstm_outputs

class PosBiLSTM(RepresentationLearner):

    def __init__(self, vocab_size, pos_vocab_size, emb_size, hidden_size, dropout, is_bidir):

        super(PosBiLSTM, self).__init__(vocab_size, emb_size)
        self.pos_vocab_size = pos_vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.is_bidir = is_bidir
        
        # See if we want to experiment with this
        # self.pos_emb_size = int(emb_size / 4)
        self.pos_emb_size = 50
        self.pos_embeddings = nn.Embedding(self.pos_vocab_size, self.pos_emb_size, padding_idx=0)
        self.pos_embeddings.weight.requires_grad = True

        self.lstm = nn.LSTM(emb_size + self.pos_emb_size, hidden_size, num_layers=1, batch_first=True, bidirectional=is_bidir)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size):
        hidden_a = torch.randn(2, batch_size, self.hidden_size)
        hidden_b = torch.randn(2, batch_size, self.hidden_size)

        if not self.is_bidir:
            hidden_a = torch.randn(1, batch_size, self.hidden_size)
            hidden_b = torch.randn(1, batch_size, self.hidden_size)

        if torch.cuda.is_available():
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, sents, pos, lengths, masks):
        embedded_sents = self.embeddings(sents)
        embedded_sents = self.dropout(embedded_sents)

        embedded_pos = self.pos_embeddings(pos)
        embedded_pos = self.dropout(embedded_pos)

        self.hidden = self.init_hidden(sents.size()[0])
        packed_sents = nn.utils.rnn.pack_padded_sequence(torch.cat([embedded_sents, embedded_pos], dim=-1), lengths, batch_first=True)
        lstm_outputs, self.hidden = self.lstm(packed_sents, self.hidden)
        lstm_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_outputs, batch_first=True)
        return lstm_outputs
