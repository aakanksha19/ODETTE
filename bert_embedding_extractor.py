import argparse
import collections
import logging
import json
import re
import pickle
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_transformers import BertTokenizer
from pytorch_transformers import BertModel

# from transformers import BertTokenizer, BertModel

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


class BertFeatureExtractor:

    def __init__(self, last_layer_index, model_path):
        if int(last_layer_index) == -1:
            self.layers = [-1]
        else:
            self.layers = list(reversed(list(range(int(last_layer_index), 0, 1))))
        # self.layers = None
        # if ',' in layers:
        #    self.layers = [int(x) for x in layers.split(',')]
        # else:
        #    self.layers = [int(layers)]
        self.model_path = model_path


    def create_examples(self, sequences):
        examples = []
        for i, sequence in enumerate(sequences):
            text_a = ' '.join(sequence).strip()
            text_b = None
            examples.append(InputExample(unique_id=i, text_a=text_a, text_b=text_b))
        return examples


    def convert_examples_to_features(self, examples, seq_length, tokenizer):
        features = []
        for i, example in enumerate(examples):
            # print(len(example.text_a.split()))
            tokens_a = tokenizer.tokenize(example.text_a)
            # print(len(tokens_a))
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                input_type_ids.append(0)  # Just indicates that this text is from input 0
            tokens.append("[SEP]")
            input_type_ids.append(0)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)
            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length
            features.append(
                    InputFeatures(
                        unique_id=example.unique_id, 
                        tokens=tokens, 
                        input_ids=input_ids, 
                        input_mask=input_mask, 
                        input_type_ids=input_type_ids
                        )
                    )
        return features


    def get_retokenized_embeds(self, input_sents, output_embeds, concat):
        all_embeds_list = []
        for sent_idx in range(len(output_embeds)):
            embeds_list = []
            features = output_embeds[sent_idx]['features'][1:-1] # Drop CLS/ SEP tokens
            # Concatenate/ sum over embeddings from all layers
            for feature in features:
                layers = feature['layers']
                if concat:
                    embed = np.concatenate([np.array(layer['values']) for layer in layers])
                else:
                    embed = np.sum([np.array(layer['values']) for layer in layers],axis=0)
                embeds_list.append(embed)

            # Mark tokens which are broken into subwords as 1,2,2,2, while marking others as 0
            tokens = [features[i]['token'] for i in range(len(features))]
            subword_list = [0 for token in tokens]
            for idx in range(len(tokens)):
                if tokens[idx].startswith('##'):
                    subword_list[idx] = 2
                    if subword_list[idx-1] != 2:
                        subword_list[idx-1] = 1

            # Construct subword lists based on aforementioned marks
            # subword_lists contains indices of full-words + first position of broken-words
            # sub_index_lists contains indices of all subwords for broken-words
            # Replace broken-words in subword_lists with the full entry from sub_index_lists
            # Fill in the indices of full-words in subword_lists
            sub_index_lists = []
            for idx in range(len(tokens)):
                if subword_list[idx] == 1:
                    curr_list = [idx]
                    idx2 = idx + 1
                    while subword_list[idx2] == 2:
                        curr_list.append(idx2)
                        if idx2 == len(tokens)-1:
                            break
                        idx2 += 1
                    sub_index_lists.append(curr_list)
            subword_list[:] = [x for x in subword_list if x != 2]
            for idx in range(len(subword_list)):
                if subword_list[idx] == 1:
                    subword_list[idx] = sub_index_lists.pop(0)
            count = 0
            for idx in range(len(subword_list)):
                if type(subword_list[idx]) == list:
                    count += len(subword_list[idx])
                else:
                    subword_list[idx] = count
                    count += 1

            final_embeds_list = []
            for idx in range(len(subword_list)):
                if type(subword_list[idx]) == list:
                    embed = np.sum([embeds_list[pos] for pos in subword_list[idx]],axis=0)/len(subword_list[idx])
                    final_embeds_list.append(embed.tolist())
                else:
                    final_embeds_list.append(embeds_list[subword_list[idx]].tolist())

            # Just verify that there is no mismatch between embeddings and tokens
            if len(input_sents[sent_idx]) != len(final_embeds_list):
                for missing_word in range(len(input_sents[sent_idx])-len(final_embeds_list)):
                    final_embeds_list.append(np.random.uniform(size=(len(embeds_list[-1].tolist()),)))
            assert len(input_sents[sent_idx]) == len(final_embeds_list)
            all_embeds_list.append(final_embeds_list)

        return all_embeds_list



    def bertify_sequences(self, input_sents, max_seq_length):
        # Convert sequences into InputExample format
        examples = self.create_examples(input_sents)

        # Load BERT Tokenizer and tokenize sequences
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BertTokenizer.from_pretrained(self.model_path, do_basic_tokenize=False)
        features = self.convert_examples_to_features(examples=examples, seq_length=max_seq_length, tokenizer=tokenizer)
        
        # Load BERT Model
        model = BertModel.from_pretrained(self.model_path, output_hidden_states=True)
        model.to(device)
        model.eval()

        # Tensorize data
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_example_index)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

        # Run BERT Model on tensorized data and collect outputs
        # of last n layers for each token, for each example
        output_embeds = []
        for input_ids, input_mask, example_indices in dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            all_encoder_layers = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            all_encoder_layers = all_encoder_layers[-1]
            for b, example_index in enumerate(example_indices):
                feature = features[example_index.item()]
                unique_id = int(feature.unique_id)
                output_bert = collections.OrderedDict()
                output_bert["linex_index"] = unique_id
                all_out_features = []
                for i, token in enumerate(feature.tokens):
                    all_layers = []
                    for j, layer_index in enumerate(self.layers):
                        layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                        layer_output = layer_output[b]
                        layers = collections.OrderedDict()
                        layers["index"] = layer_index
                        layers["values"] = [round(x.item(), 6) for x in layer_output[i]]
                        all_layers.append(layers)
                    out_features = collections.OrderedDict()
                    out_features["token"] = token
                    out_features["layers"] = all_layers
                    all_out_features.append(out_features)
                output_bert["features"] = all_out_features
                output_embeds.append(output_bert)

        bert_embeds = self.get_retokenized_embeds(input_sents, output_embeds, True)
        return bert_embeds
