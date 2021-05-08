import os
import random
from collections import Counter
import numpy as np
import torch
import copy
import corenlp
import stanfordnlp

class EventReader:

    def __init__(self):
        pass

    def read_events(self, root_dir, files):
        '''
        :param root_dir: Directory containing .tsv files with token-level event annotations
        :param files: List of files to be included in the dataset (used to pass train/dev/test splits)
        '''
        reader = open(os.path.join(root_dir, files), "r")
        file_list = []
        for line in reader:
            file_list.append(line.strip()+".tsv")
        reader.close()

        sentences = []
        events = []

        cur_sentence = []
        cur_events = []
        for file in file_list:
            reader = open(os.path.join(root_dir, file), "r")
            for line in reader:
                if len(line) > 0:
                    if line == '\n':
                        if cur_sentence:
                            sentences.append(cur_sentence)
                            events.append(cur_events)
                        cur_sentence = []
                        cur_events = []
                    else:
                        if len(line.strip().split('\t')) == 1:
                            continue
                        word, event = line.strip().split('\t')
                        cur_sentence.append(word.lower())
                        cur_events.append(event)
            if cur_sentence:
                sentences.append(cur_sentence)
                events.append(cur_events)

        return sentences, events

    def create_padded_batches(self, sentences, events, batch_size, use_cuda, shuffle, is_bert=False, inst_weights=None):
        combined = list(zip(sentences, events))
        if inst_weights is not None:
            combined = list(zip(sentences, events, inst_weights))
        if shuffle:
            random.shuffle(combined)
        shuffled_sents, shuffled_events = [], []
        if inst_weights is not None:
            shuffled_sents, shuffled_events, shuffled_weights = zip(*combined)
            shuffled_weights = list(shuffled_weights)
        else:
            shuffled_sents, shuffled_events = zip(*combined)
        shuffled_events = list(shuffled_events)
        shuffled_sents = list(shuffled_sents)

        batches = []

        for i in range(0, len(shuffled_sents), batch_size):
            start = i
            end = min(len(shuffled_sents), start+batch_size)
            cur_sents = shuffled_sents[start:end]
            cur_events = shuffled_events[start:end]
            cur_weights = None
            if inst_weights is not None:
                cur_weights = shuffled_weights[start:end]

            if cur_weights is not None:
                combined = list(zip(cur_sents, cur_events, cur_weights))
            else:
                combined = list(zip(cur_sents, cur_events))
            combined = list(reversed(sorted(combined, key=lambda x: len(x[0]))))
            if cur_weights is not None:
                cur_sents, cur_events, cur_weights = zip(*combined)
                cur_weights = list(cur_weights)
            else:
                cur_sents, cur_events = zip(*combined)
            cur_sents = list(cur_sents)
            cur_events = list(cur_events)
            cur_lengths = [len(x) for x in cur_sents]
            cur_masks = []

            max_seq_len = cur_lengths[0]
            for i in range(len(cur_sents)):
                if not is_bert:
                    cur_sents[i] = cur_sents[i] + [0] * (max_seq_len - cur_lengths[i])
                else:
                    cur_sents[i] = cur_sents[i] + [[0] * len(cur_sents[i][0]) for j in range(max_seq_len - cur_lengths[i])]
                cur_events[i] = cur_events[i] + [0] * (max_seq_len - cur_lengths[i])
                cur_masks.append([1] * cur_lengths[i] + [0] * (max_seq_len - cur_lengths[i]))

            if not is_bert:
                if not use_cuda:
                    if cur_weights is not None:
                        batches.append([torch.LongTensor(cur_sents), torch.FloatTensor(cur_events), torch.LongTensor(cur_lengths), torch.FloatTensor(cur_masks), torch.FloatTensor(cur_weights)])
                    else:
                        batches.append([torch.LongTensor(cur_sents), torch.FloatTensor(cur_events), torch.LongTensor(cur_lengths), torch.FloatTensor(cur_masks)])
                else:
                    if cur_weights is not None:
                        batches.append([torch.cuda.LongTensor(cur_sents), torch.cuda.FloatTensor(cur_events), torch.cuda.LongTensor(cur_lengths), torch.cuda.FloatTensor(cur_masks), torch.cuda.FloatTensor(cur_weights)])
                    else:
                        batches.append([torch.cuda.LongTensor(cur_sents), torch.cuda.FloatTensor(cur_events), torch.cuda.LongTensor(cur_lengths), torch.cuda.FloatTensor(cur_masks)])
            else:
                if not use_cuda:
                    if cur_weights is not None:
                        batches.append([torch.FloatTensor(cur_sents), torch.FloatTensor(cur_events), torch.LongTensor(cur_lengths), torch.FloatTensor(cur_masks), torch.FloatTensor(cur_weights)])
                    else:
                        batches.append([torch.FloatTensor(cur_sents), torch.FloatTensor(cur_events), torch.LongTensor(cur_lengths), torch.FloatTensor(cur_masks)])
                else:
                    #for i in range(len(cur_masks)):
                    #    print('{} {} {}'.format(len(cur_sents[i]), len(cur_events[i]), len(cur_masks[i])))
                    if cur_weights is not None:
                        batches.append([torch.cuda.FloatTensor(cur_sents), torch.cuda.FloatTensor(cur_events), torch.cuda.LongTensor(cur_lengths), torch.cuda.FloatTensor(cur_masks), torch.cuda.FloatTensor(cur_weights)])
                    else:
                        batches.append([torch.cuda.FloatTensor(cur_sents), torch.cuda.FloatTensor(cur_events), torch.cuda.LongTensor(cur_lengths), torch.cuda.FloatTensor(cur_masks)])

        return batches

    def create_pos_padded_batches(self, sentences, pos, events, batch_size, use_cuda, shuffle):
        combined = list(zip(sentences, pos, events))
        if shuffle:
            random.shuffle(combined)
        shuffled_sents, shuffled_pos, shuffled_events = zip(*combined)
        shuffled_events = list(shuffled_events)
        shuffled_pos = list(shuffled_pos)
        shuffled_sents = list(shuffled_sents)

        batches = []
        for i in range(0, len(shuffled_sents), batch_size):
            start = i
            end = min(len(shuffled_sents), start+batch_size)
            cur_sents = shuffled_sents[start:end]
            cur_pos = shuffled_pos[start:end]
            cur_events = shuffled_events[start:end]

            combined = list(zip(cur_sents, cur_pos, cur_events))
            combined = list(reversed(sorted(combined, key=lambda x: len(x[0]))))
            cur_sents, cur_pos, cur_events = zip(*combined)
            cur_sents = list(cur_sents)
            cur_pos = list(cur_pos)
            cur_events = list(cur_events)
            cur_lengths = [len(x) for x in cur_sents]
            cur_masks = []

            max_seq_len = cur_lengths[0]

            for i in range(len(cur_sents)):
                cur_sents[i] = cur_sents[i] + [0] * (max_seq_len - cur_lengths[i])
                cur_pos[i] = cur_pos[i] + [0] * (max_seq_len - cur_lengths[i])
                cur_events[i] = cur_events[i] + [0] * (max_seq_len - cur_lengths[i])
                cur_masks.append([1] * cur_lengths[i] + [0] * (max_seq_len - cur_lengths[i]))

            if not use_cuda:
                batches.append([torch.LongTensor(cur_sents), torch.LongTensor(cur_pos), torch.FloatTensor(cur_events), torch.LongTensor(cur_lengths), torch.FloatTensor(cur_masks)])
            else:
                batches.append([torch.cuda.LongTensor(cur_sents), torch.cuda.LongTensor(cur_pos), torch.cuda.FloatTensor(cur_events), torch.cuda.LongTensor(cur_lengths), torch.cuda.FloatTensor(cur_masks)])

        return batches

    def construct_vocab(self, sequences):
        counter = Counter()
        for sequence in sequences:
            counter.update(sequence)
        vocab = ["<PAD>", "<UNK>"] + list(counter.keys())
        print(len(vocab))
        vocab = dict(list(zip(vocab, range(len(vocab)))))
        return vocab

    def construct_integer_sequences(self, sequences, vocab):
        int_sequences = []
        for sequence in sequences:
            new_sequence = []
            for word in sequence:
                if word in vocab:
                    new_sequence.append(vocab[word])
                else:
                    if "<UNK>" in vocab:
                        new_sequence.append(vocab["<UNK>"])
                    else:
                        new_sequence.append(vocab["<unk>"])   # For compatibility of this function with models where unk is lowercase
            int_sequences.append(new_sequence)
        return int_sequences


# Write a sentence-reader class to read sentences with their domain
class SentenceReader:

    def __init__(self):
        pass

    def read_unlabeled_sents(self, root_dir):
        sentences = []
        domains = []
        for file in os.listdir(root_dir):
            reader = open(os.path.join(root_dir, file), "r")
            for line in reader:
                if line == '\n':
                    continue
                sentences.append([x.lower() for x in line.strip().split()])
                domains.append(0)
        return sentences, domains

    def read_unlabeled_sents_as_docs(self, root_dir):
        files = []
        filenames = []
        for file in os.listdir(root_dir):
            filenames.append(file)
            cur_file = []
            reader = open(os.path.join(root_dir, file), "r")
            for line in reader:
                if line == "\n":
                    continue
                cur_file.append([x.lower() for x in line.strip().split()])
            files.append(cur_file)
            reader.close()
        return filenames, files

    def read_labeled_sents(self, sents):
        new_sents = copy.deepcopy(sents)
        domains = [1]*len(new_sents)
        return new_sents, domains

    def create_padded_batches(self, sentences, domains, batch_size, use_cuda, shuffle, is_bert=False):
        combined = list(zip(sentences, domains))
        if shuffle:
            random.shuffle(combined)
        shuffled_sents, shuffled_domains = zip(*combined)
        shuffled_domains = list(shuffled_domains)
        shuffled_sents = list(shuffled_sents)

        batches = []
        for i in range(0, len(shuffled_sents), batch_size):
            start = i
            end = min(len(shuffled_sents), start + batch_size)
            cur_sents = shuffled_sents[start:end]
            cur_domains = shuffled_domains[start:end]

            combined = list(zip(cur_sents, cur_domains))
            combined = list(reversed(sorted(combined, key=lambda x: len(x[0]))))
            cur_sents, cur_domains = zip(*combined)
            cur_sents = list(cur_sents)
            cur_domains = list(cur_domains)
            cur_lengths = [len(x) for x in cur_sents]
            cur_masks = []

            max_seq_len = cur_lengths[0]

            for i in range(len(cur_sents)):
                if not is_bert:
                    cur_sents[i] = cur_sents[i] + [0] * (max_seq_len - cur_lengths[i])
                else:
                    cur_sents[i] = cur_sents[i] + [[0] * len(cur_sents[i][0]) for j in range(max_seq_len - cur_lengths[i])]
                cur_masks.append([1] * cur_lengths[i] + [0] * (max_seq_len - cur_lengths[i]))

            if not is_bert:
                if not use_cuda:
                    batches.append(
                        [torch.LongTensor(cur_sents), torch.LongTensor(cur_domains), torch.LongTensor(cur_lengths),
                        torch.FloatTensor(cur_masks)])
                else:
                    batches.append([torch.cuda.LongTensor(cur_sents), torch.cuda.LongTensor(cur_domains),
                                torch.cuda.LongTensor(cur_lengths), torch.cuda.FloatTensor(cur_masks)])
            else:
                if not use_cuda:
                    batches.append(
                        [torch.FloatTensor(cur_sents), torch.LongTensor(cur_domains), torch.LongTensor(cur_lengths),
                        torch.FloatTensor(cur_masks)])
                else:
                    batches.append([torch.cuda.FloatTensor(cur_sents), torch.cuda.LongTensor(cur_domains),
                                torch.cuda.LongTensor(cur_lengths), torch.cuda.FloatTensor(cur_masks)])

        return batches

    def create_pos_padded_batches(self, sentences, pos, domains, batch_size, use_cuda, shuffle):
        combined = list(zip(sentences, pos, domains))
        if shuffle:
            random.shuffle(combined)
        shuffled_sents, shuffled_pos, shuffled_domains = zip(*combined)
        shuffled_domains = list(shuffled_domains)
        shuffled_pos = list(shuffled_pos)
        shuffled_sents = list(shuffled_sents)

        batches = []
        for i in range(0, len(shuffled_sents), batch_size):
            start = i
            end = min(len(shuffled_sents), start + batch_size)
            cur_sents = shuffled_sents[start:end]
            cur_pos = shuffled_pos[start:end]
            cur_domains = shuffled_domains[start:end]

            combined = list(zip(cur_sents, cur_pos, cur_domains))
            combined = list(reversed(sorted(combined, key=lambda x: len(x[0]))))
            cur_sents, cur_pos, cur_domains = zip(*combined)
            cur_sents = list(cur_sents)
            cur_pos = list(cur_pos)
            cur_domains = list(cur_domains)
            cur_lengths = [len(x) for x in cur_sents]
            cur_masks = []

            max_seq_len = cur_lengths[0]

            for i in range(len(cur_sents)):
                cur_sents[i] = cur_sents[i] + [0] * (max_seq_len - cur_lengths[i])
                cur_pos[i] = cur_pos[i] + [0] * (max_seq_len - cur_lengths[i])
                cur_masks.append([1] * cur_lengths[i] + [0] * (max_seq_len - cur_lengths[i]))

            if not use_cuda:
                batches.append(
                    [torch.LongTensor(cur_sents), torch.LongTensor(cur_pos), torch.LongTensor(cur_domains), torch.LongTensor(cur_lengths),
                     torch.FloatTensor(cur_masks)])
            else:
                batches.append([torch.cuda.LongTensor(cur_sents), torch.cuda.LongTensor(cur_pos), torch.cuda.LongTensor(cur_domains),
                                torch.cuda.LongTensor(cur_lengths), torch.cuda.FloatTensor(cur_masks)])

        return batches

# TODO: Replace models_dir with local directory containing Stanford CoreNLP English model download
class Parser:
    
    def __init__(self):
        self.pipeline = stanfordnlp.Pipeline(processors='tokenize,pos', lang='en', tokenize_pretokenized=True, models_dir="/usr0/home/anaik/installations/stanford-corenlp-full-2018-10-05/en/", treebank="en_ewt")

    def parse_sequences(self, sequences):
        parse_outputs = []
        for sequence in sequences:
            annotations = self.pipeline([sequence])
            pos_tags = []
            for sentence in annotations.sentences:
                for token in sentence.words:
                    pos_tags.append(token.pos)
            parse_outputs.append(pos_tags)
        return parse_outputs
