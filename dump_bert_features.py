import random
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from event_dataset import EventReader, SentenceReader, Parser
from da_models_new import AdversarialEventExtractor, GradReverse
from bert_embedding_extractor import BertFeatureExtractor

# Change train function to do alternating optimization
def train(model, train_batches, dev_batches, adv_batches, num_epochs, learning_rate, use_cuda, path):

    event_criterion = nn.BCEWithLogitsLoss()
    adv_criterion = nn.CrossEntropyLoss()

    adv_step_optimizer = optim.Adam(model.adv_classifier.parameters(), lr=learning_rate)
    event_step_optimizer = optim.Adam(model.event_extractor.parameters(), lr=learning_rate)

    best_precision, best_recall, best_f1 = 0.0, 0.0, 0.0

    for epoch in range(num_epochs):
        total_event_loss = 0.0
        total_adv_loss = 0.0

        random.shuffle(adv_batches)
        num_batches = len(train_batches)

        for i, batch in enumerate(train_batches):
            batch = [x.to('cuda') for x in batch]
            adv_batch = [x.to('cuda') for x in adv_batches[i]]

            adv_step_optimizer.zero_grad()
            event_step_optimizer.zero_grad()
            domain_outputs, event_outputs, event_domains = model(batch, adv_batch)

            # Optimize adversarial classifier
            adv_labels = adv_batch[-3]
            adv_loss = adv_criterion(domain_outputs, adv_labels)
            total_adv_loss += adv_loss.item()
            adv_loss.backward()
            adv_step_optimizer.step()

            # Flush out gradients and compute second loss over events
            
            adv_step_optimizer.zero_grad()
            event_step_optimizer.zero_grad()

            event_labels = batch[-3].contiguous().view(-1,1)
            dom_labels = torch.ones(batch[0].size()[0], dtype=torch.int64)
            if use_cuda:
                dom_labels = dom_labels.cuda()
            event_loss = adv_criterion(event_domains, dom_labels) + event_criterion(event_outputs, event_labels)
            total_event_loss += event_loss.item()
            event_loss.backward()
            event_step_optimizer.step()

        total_adv_loss /= num_batches
        total_event_loss /= num_batches
        print("Adversarial Loss at epoch {}: {}".format(epoch, total_adv_loss))
        print("Event Loss at epoch {}: {}".format(epoch, total_event_loss))
        print("Performance on development set:")
        precision, recall, f1 = test(model, dev_batches, use_cuda, '')
        if f1 > best_f1:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            torch.save(model.state_dict(), path)
        model.train()


def test(model, dev_batches, use_cuda, path):
    if path != '':
        model.load_state_dict(torch.load(path))
    model.eval()
    predicted, gold, correct = 0.0, 0.0, 0.0
    domain_acc = 0.0
    for batch in dev_batches:
        batch = [x.to('cuda') for x in batch]
        labels = batch[-3]
        labels = labels.contiguous().view(-1, 1)
        domain_outputs, event_outputs, event_domains = model(batch, batch)
        _, event_domain_outputs = torch.max(event_domains, dim=1)
        if use_cuda:
            event_outputs = event_outputs.cpu().detach().numpy()
            event_domain_outputs = event_domain_outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
        else:
            event_domain_outputs = event_domain_outputs.numpy()
        domain_acc += np.sum(event_domain_outputs == np.ones(event_domain_outputs.shape[0])) / event_domain_outputs.shape[0]
        cur_correct, cur_pred, cur_gold = calculate_batch_f1(event_outputs.tolist(), labels.tolist())
        predicted += cur_pred
        gold += cur_gold
        correct += cur_correct
    precision = correct / predicted if predicted != 0 else 0.0
    recall = correct / gold if gold != 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0.0
    domain_acc /= len(dev_batches)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 Score: {}".format(f1))
    print("Domain Prediction Accuracy: {}".format(domain_acc))
    return precision, recall, f1

def calculate_batch_f1(preds, labels):
    predicted = 0.0
    gold = 0.0
    correct = 0.0
    for pred, label in zip(preds, labels):
        pred = 0 if pred[0] <= 0.0 else 1
        label = label[0]
        if pred == 1:
            predicted += 1
        if label == 1:
            gold += 1
        if pred == label and label == 1:
            correct += 1
    return correct, predicted, gold

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", action="store", type=str, required=True, help="Directory containing source labeled data files")
    parser.add_argument("--target_dir", action="store", type=str, default=None, help="Directory containing target data files")
    parser.add_argument("--train_file", action="store", type=str, required=True, help="File containing list of train documents")
    parser.add_argument("--dev_file", action="store", type=str, required=True, help="File containing list of dev documents")
    parser.add_argument("--test_file", action="store", type=str, required=True, help="File containing list of test documents")
    parser.add_argument("--save_path", action="store", type=str, required=True, help="Path to store feature batches created using BERT")
    parser.add_argument("--suffix", action="store", type=str, required=True, help="Suffix to store feature batches created using BERT")
    parser.add_argument("--bert_model_path", action="store", type=str, default="bert-base-uncased", help="Path to BERT model used for embedding construction")
    parser.add_argument("--emb_file", action="store", type=str, default=None, help="Path to pretrained embedding file")
    parser.add_argument("--batch_size", action="store", type=int, default=16, help="Batch size")
    parser.add_argument("--emb_size", action="store", type=int, default=100, help="Embedding size")
    parser.add_argument("--seed", action="store", type=int, default=0, help="Random seed")
    parser.add_argument("--model", action="store", type=str, default="word", help="Specify type of features to be used in model")
    parser.add_argument("--emb_layers", action="store", type=str, default=-4, help="Specify which BERT layers to use")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    reader = EventReader()
    parser = Parser()

    train_sentences, train_events = reader.read_events(args.data_dir, args.train_file)
    dev_sentences, dev_events = reader.read_events(args.data_dir, args.dev_file)
    test_sentences, test_events = reader.read_events(args.data_dir, args.test_file)
    
    # train_sentences, train_events = train_sentences[:50], train_events[:50]
    # dev_sentences, dev_events = dev_sentences[:50], dev_events[:50]
    # test_sentences, test_events = test_sentences[:50], test_events[:50]

    # train_parse = parser.parse_sequences(train_sentences)
    # dev_parse = parser.parse_sequences(dev_sentences)
    # test_parse = parser.parse_sequences(test_sentences)

    # Read in new-domain data, create batches and construct vocab over that
    unlabeled_sents, unlabeled_domains, labeled_sents, labeled_domains = [], [], [], []
    if args.target_dir is not None:
        sent_reader = SentenceReader()
        unlabeled_sents, unlabeled_domains = sent_reader.read_unlabeled_sents(args.target_dir)
        labeled_sents, labeled_domains = sent_reader.read_labeled_sents(train_sentences)
    '''
    # unlabeled_sents = unlabeled_sents[:50]
    # unlabeled_domains = unlabeled_domains[:50]

    # Parse raw sentences
    labeled_parse = parser.parse_sequences(labeled_sents)
    unlabeled_parse = parser.parse_sequences(unlabeled_sents)

    combined = list(zip(unlabeled_parse, unlabeled_sents))
    random.shuffle(combined)
    unlabeled_parse, unlabeled_sents = zip(*combined)
    unlabeled_parse = list(unlabeled_parse)
    unlabeled_sents = list(unlabeled_sents)
    adv_sents = unlabeled_sents[:len(labeled_sents)] + labeled_sents
    adv_parse = unlabeled_parse[:len(labeled_sents)] + labeled_parse
    adv_domains = unlabeled_domains[:len(labeled_sents)] + labeled_domains

    sent_vocab = reader.construct_vocab(train_sentences + dev_sentences + test_sentences + unlabeled_sents)
    pos_vocab = reader.construct_vocab(train_parse + dev_parse + test_parse + unlabeled_parse)
    label_vocab = {"O": 0, "EVENT": 1}
    use_shared_vocab = False

    if args.do_train:
        pickle.dump(pos_vocab, open(args.model_path+"_posvocab_{}.pkl".format(args.seed), "wb"))
        if not use_shared_vocab:
            pickle.dump(sent_vocab, open(args.model_path+"_vocab_{}.pkl".format(args.seed), "wb"))
        else:
            sent_vocab = pickle.load(open("../models/abridge_aug_vocab.pkl".format(args.seed), "rb"))
    elif args.do_eval:
        pos_vocab = pickle.load(open(args.model_path+"_posvocab_{}.pkl".format(args.seed), "rb"))
        if not use_shared_vocab:
            sent_vocab = pickle.load(open(args.model_path+"_vocab_{}.pkl".format(args.seed), "rb"))
        else:
            sent_vocab = pickle.load(open("../models/abridge_aug_vocab.pkl".format(args.seed), "rb"))
    '''
    label_vocab = {"O": 0, "EVENT": 1, "ENT": 0}
    adv_sents, adv_domains = [], []
    if args.target_dir is not None:
        random.shuffle(unlabeled_sents)
        adv_sents = unlabeled_sents[:len(labeled_sents)] + labeled_sents
        adv_domains = unlabeled_domains[:len(labeled_sents)] + labeled_domains
    # int_train_sents = reader.construct_integer_sequences(train_sentences, sent_vocab)
    int_train_labels = reader.construct_integer_sequences(train_events, label_vocab)
    # int_dev_sents = reader.construct_integer_sequences(dev_sentences, sent_vocab)
    int_dev_labels = reader.construct_integer_sequences(dev_events, label_vocab)
    # int_test_sents = reader.construct_integer_sequences(test_sentences, sent_vocab)
    int_test_labels = reader.construct_integer_sequences(test_events, label_vocab)
    # int_train_parse = reader.construct_integer_sequences(train_parse, pos_vocab)
    # int_dev_parse = reader.construct_integer_sequences(dev_parse, pos_vocab)
    # int_test_parse = reader.construct_integer_sequences(test_parse, pos_vocab)

    # int_adv_sents = reader.construct_integer_sequences(adv_sents, sent_vocab)
    # int_adv_parse = reader.construct_integer_sequences(adv_parse, pos_vocab)

    train_batches, dev_batches, test_batches, adv_batches = [], [], [], []

    if args.model == "word":
        train_batches = reader.create_padded_batches(int_train_sents, int_train_labels, args.batch_size, use_cuda, True)
        dev_batches = reader.create_padded_batches(int_dev_sents, int_dev_labels, args.batch_size, use_cuda, False)
        test_batches = reader.create_padded_batches(int_test_sents, int_test_labels, args.batch_size, use_cuda, False)

        adv_batches = sent_reader.create_padded_batches(int_adv_sents, adv_domains, args.batch_size, use_cuda, True)

    elif args.model == "pos":
        train_batches = reader.create_pos_padded_batches(int_train_sents, int_train_parse, int_train_labels, args.batch_size, use_cuda, True)
        dev_batches = reader.create_pos_padded_batches(int_dev_sents, int_dev_parse, int_dev_labels, args.batch_size, use_cuda, False)
        test_batches = reader.create_pos_padded_batches(int_test_sents, int_test_parse, int_test_labels, args.batch_size, use_cuda, False)

        adv_batches = sent_reader.create_pos_padded_batches(int_adv_sents, int_adv_parse, adv_domains, args.batch_size, use_cuda, True)

    elif args.model == "bert":
        feature_extractor = BertFeatureExtractor(args.emb_layers, args.bert_model_path)
        
        sent_berts = feature_extractor.bertify_sequences(train_sentences, max_seq_length=450)
        batches = reader.create_padded_batches(sent_berts, int_train_labels, args.batch_size, use_cuda, True, True)
        pickle.dump(batches, open(os.path.join(args.save_path, "bert_train_batches_{}.pkl".format(args.suffix)), "wb"))
        print('Dumped training data')

        sent_berts = feature_extractor.bertify_sequences(dev_sentences, max_seq_length=450)
        batches = reader.create_padded_batches(sent_berts, int_dev_labels, args.batch_size, use_cuda, False, True)
        pickle.dump(batches, open(os.path.join(args.save_path, "bert_dev_batches_{}.pkl".format(args.suffix)), "wb"))
        print('Dumped development data')

        sent_berts = feature_extractor.bertify_sequences(test_sentences, max_seq_length=450)
        batches = reader.create_padded_batches(sent_berts, int_test_labels, args.batch_size, use_cuda, False, True)
        pickle.dump(batches, open(os.path.join(args.save_path, "bert_test_batches_{}.pkl".format(args.suffix)), "wb"))
        print('Dumped test data')
        
        if args.target_dir is not None:
            sent_berts = feature_extractor.bertify_sequences(adv_sents, max_seq_length=450)
            batches = sent_reader.create_padded_batches(sent_berts, adv_domains, args.batch_size, use_cuda, True, True)
            pickle.dump(batches, open(os.path.join(args.save_path, "bert_adv_batches_{}.pkl".format(args.suffix)), "wb"))
            print('Dumped unlabeled data')
'''
        train_batches = reader.create_padded_batches(train_sent_berts, int_train_labels, args.batch_size, use_cuda, True, True)
        dev_batches = reader.create_padded_batches(dev_sent_berts, int_dev_labels, args.batch_size, use_cuda, False, True)
        test_batches = reader.create_padded_batches(test_sent_berts, int_test_labels, args.batch_size, use_cuda, False, True)
        adv_batches = sent_reader.create_padded_batches(adv_sent_berts, adv_domains, args.batch_size, use_cuda, True, True)
        
        pickle.dump(train_batches, open(os.path.join(args.save_path, "bert_train_batches_{}.pkl".format(args.suffix)), "wb"))
        pickle.dump(dev_batches, open(os.path.join(args.save_path, "bert_dev_batches_{}.pkl".format(args.suffix)), "wb"))
        pickle.dump(test_batches, open(os.path.join(args.save_path, "bert_test_batches_{}.pkl".format(args.suffix)), "wb"))
        pickle.dump(adv_batches, open(os.path.join(args.save_path, "bert_adv_batches_{}.pkl".format(args.suffix)), "wb"))

    if args.model == 'word':
        model = AdversarialEventExtractor(len(list(sent_vocab.keys())), args.emb_size, args.hidden_size, 1, args.adv_size, args.adv_layers, args.num_domains, args.dropout, args.bidir, args.model)
    elif args.model == 'pos':
        model = AdversarialEventExtractor(len(list(sent_vocab.keys())), args.emb_size, args.hidden_size, 1, args.adv_size, args.adv_layers, args.num_domains, args.dropout, args.bidir, args.model, len(list(pos_vocab.keys())))
    elif args.model == 'bert':
        print('Embedding size: {}'.format(train_batches[0][0].size()[-1]))
        model = AdversarialEventExtractor(len(list(sent_vocab.keys())), train_batches[0][0].size()[-1], args.hidden_size, 1, args.adv_size, args.adv_layers, args.num_domains, args.dropout, args.bidir, args.model)
    if args.emb_file is not None:
        model.event_extractor.rep_learner.load_embeddings(args.emb_file, sent_vocab)
    if use_cuda:
        model = model.cuda()


    if args.do_train:
        train(model, train_batches, dev_batches, adv_batches, args.num_epochs, args.learning_rate, use_cuda, args.model_path+"_{}.pth".format(args.seed))
        if args.do_eval:
            test(model, test_batches, use_cuda, args.model_path+"_{}.pth".format(args.seed))
    else:
        test(model, test_batches, use_cuda, args.model_path+"_{}.pth".format(args.seed))
'''
