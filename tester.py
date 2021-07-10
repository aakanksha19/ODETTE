import random
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from event_dataset import EventReader, Parser
from word_models import EventExtractor
from bert_embedding_extractor import BertFeatureExtractor

def train(model, train_batches, dev_batches, num_epochs, learning_rate, use_cuda, path, model_type):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(list(model.rep_learner.parameters()) + list(model.classifier.parameters()), lr=learning_rate)
    best_precision, best_recall, best_f1 = 0.0, 0.0, 0.0
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = len(train_batches)
        for batch in train_batches:
            optimizer.zero_grad()
            batch = [x.to('cuda') for x in batch]
            labels, predictions = batch[-3], []
            labels = labels.contiguous().view(-1,1)
            if model_type == "word" or model_type == "delex" or model_type.startswith("bert"):
                sents, _, lengths, masks = batch
                predictions = model(sents, lengths, masks)
            elif model_type == "pos":
                sents, pos, _, lengths, masks = batch
                predictions = model(sents, lengths, masks, pos)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        total_loss /= num_batches
        print("Training Loss at epoch {}: {}".format(epoch, total_loss))
        print("Performance on development set:")
        precision, recall, f1 = test(model, dev_batches, use_cuda, '', args.model)
        if f1 > best_f1:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            torch.save(model.state_dict(), path)
        model.train()

def test(model, dev_batches, use_cuda, path, model_type, oov=None):
    if path != '':
        model.load_state_dict(torch.load(path))
    model.eval()
    predicted, gold, correct = 0.0, 0.0, 0.0
    # all_reps = []
    iv_predicted, iv_gold, iv_correct = 0.0, 0.0, 0.0
    oov_predicted, oov_gold, oov_correct = 0.0, 0.0, 0.0
    for batch in dev_batches:
        cpu_sents = batch[0].view(-1,1)
        batch = [x.to('cuda') for x in batch]
        labels, predictions = batch[-3], []
        labels = labels.contiguous().view(-1,1)
        if model_type == "word" or model_type == "delex" or model_type.startswith("bert"):
            sents, _, lengths, masks = batch
            predictions = model(sents, lengths, masks)   # Remove reps after dumping BERT
        elif model_type == "pos":
            sents, pos, _, lengths, masks = batch
            predictions = model(sents, lengths, masks, pos)
        if use_cuda:
            predictions = predictions.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            # reps = reps.cpu()  # Comment out after dumping BERT
        cur_correct, cur_pred, cur_gold = calculate_batch_f1(predictions.tolist(), labels.tolist())
        if oov is not None:
            cur_iv_correct, cur_iv_pred, cur_iv_gold, cur_oov_correct, cur_oov_pred, cur_oov_gold = calculate_split_f1(predictions.tolist(), labels.tolist(), oov, cpu_sents, sent_vocab)
            iv_correct += cur_iv_correct
            iv_gold += cur_iv_gold
            iv_predicted += cur_iv_pred
            oov_correct += cur_oov_correct
            oov_gold += cur_oov_gold
            oov_predicted += cur_oov_pred
        predicted += cur_pred
        gold += cur_gold
        correct += cur_correct
        # all_reps.append(reps)
    # pickle.dump(all_reps, open('BERT_reps_rec.pkl', 'wb'))
    # print('Dumped BERT record reps')
    precision = correct / predicted if predicted != 0 else 0.0
    recall = correct / gold if gold != 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0.0
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 Score: {}".format(f1))
    return precision, recall, f1

def calculate_split_f1(preds, labels, oov, raw_sents, sent_vocab):
    iv_predicted, iv_gold, iv_correct = 0.0, 0.0, 0.0
    oov_predicted, oov_gold, oov_correct = 0.0, 0.0, 0.0
    raw_sents = raw_sents.tolist()
    # print(len(raw_sents))
    # exit(1)
    i = 0
    for pred, label in zip(preds, labels):
        pred = 0 if pred[0] <= 0.0 else 1
        label = label[0]
        cur_words = raw_sents[i]
        #if pred == 1:
        i += 1

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
    parser.add_argument("--data_dir", action="store", type=str, required=True, help="Directory containing data files")
    parser.add_argument("--train_file", action="store", type=str, required=True, help="File containing list of train documents")
    parser.add_argument("--dev_file", action="store", type=str, required=True, help="File containing list of dev documents")
    parser.add_argument("--test_file", action="store", type=str, required=True, help="File containing list of test documents")
    parser.add_argument("--model_path", action="store", type=str, required=True, help="Path to store/ load trained model")
    parser.add_argument("--emb_file", action="store", type=str, default=None, help="Path to pretrained embedding file")
    parser.add_argument("--batch_size", action="store", type=int, default=16, help="Batch size")
    parser.add_argument("--emb_size", action="store", type=int, default=100, help="Embedding size")
    parser.add_argument("--hidden_size", action="store", type=int, default=100, help="Hidden size")
    parser.add_argument("--dropout", action="store", type=float, default=0.5, help="Dropout")
    parser.add_argument("--num_epochs", action="store", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--learning_rate", action="store", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--bidir", action="store_false", default=True, help="Specify whether LSTM should be bidirectional")
    parser.add_argument("--seed", action="store", type=int, default=0, help="Random seed")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--model", action="store", type=str, default="word", help="Specify type of features to use for model")
    parser.add_argument("--save_path", action="store", type=str, default=None, help="Path to load BERT representations from")
    parser.add_argument("--suffix", action="store", type=str, default=None, help="Dataset name suffix")
    parser.add_argument("--num_layers", action="store", type=int, default=3, help="Specify the number of layers to use for MLP")
    parser.add_argument("--oov_vocab", action="store", type=str, default=None, help="Specify path to OOV vocab to split evaluation")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = torch.cuda.is_available()

    reader = EventReader()
    parser = Parser()
    train_sentences, train_events = reader.read_events(args.data_dir, args.train_file)
    dev_sentences, dev_events = reader.read_events(args.data_dir, args.dev_file)
    test_sentences, test_events = reader.read_events(args.data_dir, args.test_file)

    # train_sentences, train_events = train_sentences[:50], train_events[:50]
    # dev_sentences, dev_events = dev_sentences[:50], dev_events[:50]
    # test_sentences, test_events = test_sentences[:50], test_events[:50]

    train_parse = parser.parse_sequences(train_sentences)
    dev_parse = parser.parse_sequences(dev_sentences)
    test_parse = parser.parse_sequences(test_sentences)

    sent_vocab = reader.construct_vocab(train_sentences + dev_sentences + test_sentences)
    pos_vocab = reader.construct_vocab(train_parse + dev_parse + test_parse)
    label_vocab = {"O": 0, "EVENT": 1, "ENT": 0}
    use_shared_vocab = True

    oov_vocab = None
    if args.oov_vocab is not None:
        pickle.load(open(args.oov_vocab, 'rb'))

    if args.do_train:
        pickle.dump(pos_vocab, open(args.model_path+"_posvocab_{}.pkl".format(args.seed), "wb"))
        if use_shared_vocab:
            sent_vocab = pickle.load(open("shared_vocab_news_lit.pkl", "rb"))
        pickle.dump(sent_vocab, open(args.model_path+"_vocab_{}.pkl".format(args.seed), "wb"))
    elif args.do_eval:
        pos_vocab = pickle.load(open(args.model_path+"_posvocab_{}.pkl".format(args.seed), "rb"))
        sent_vocab = pickle.load(open(args.model_path+"_vocab_{}.pkl".format(args.seed), "rb"))

    int_train_sents = reader.construct_integer_sequences(train_sentences, sent_vocab)
    int_train_labels = reader.construct_integer_sequences(train_events, label_vocab)
    int_dev_sents = reader.construct_integer_sequences(dev_sentences, sent_vocab)
    int_dev_labels = reader.construct_integer_sequences(dev_events, label_vocab)
    int_test_sents = reader.construct_integer_sequences(test_sentences, sent_vocab)
    int_test_labels = reader.construct_integer_sequences(test_events, label_vocab)
    int_train_parse = reader.construct_integer_sequences(train_parse, pos_vocab)
    int_dev_parse = reader.construct_integer_sequences(dev_parse, pos_vocab)
    int_test_parse = reader.construct_integer_sequences(test_parse, pos_vocab)

    train_batches, dev_batches, test_batches = [], [], []

    if args.model == "word":
        train_batches = reader.create_padded_batches(int_train_sents, int_train_labels, args.batch_size, use_cuda, True)
        dev_batches = reader.create_padded_batches(int_dev_sents, int_dev_labels, args.batch_size, use_cuda, False)
        test_batches = reader.create_padded_batches(int_test_sents, int_test_labels, args.batch_size, use_cuda, False)

    elif args.model == "delex":
        train_batches = reader.create_padded_batches(int_train_parse, int_train_labels, args.batch_size, use_cuda, True)
        dev_batches = reader.create_padded_batches(int_dev_parse, int_dev_labels, args.batch_size, use_cuda, False)
        test_batches = reader.create_padded_batches(int_test_parse, int_test_labels, args.batch_size, use_cuda, False)

    elif args.model == "pos":
        train_batches = reader.create_pos_padded_batches(int_train_sents, int_train_parse, int_train_labels, args.batch_size, use_cuda, True)
        dev_batches = reader.create_pos_padded_batches(int_dev_sents, int_dev_parse, int_dev_labels, args.batch_size, use_cuda, False)
        test_batches = reader.create_pos_padded_batches(int_test_sents, int_test_parse, int_test_labels, args.batch_size, use_cuda, False)
    elif args.model.startswith("bert"):
        # feature_extractor = BertFeatureExtractor("-1,-2,-3,-4")
        # train_sent_berts = feature_extractor.bertify_sequences(train_sentences, max_seq_length=450)
        # dev_sent_berts = feature_extractor.bertify_sequences(dev_sentences, max_seq_length=450)
        # test_sent_berts = feature_extractor.bertify_sequences(test_sentences, max_seq_length=450)
        
        # train_batches = reader.create_padded_batches(train_sent_berts, int_train_labels, args.batch_size, use_cuda, True, True)
        # dev_batches = reader.create_padded_batches(dev_sent_berts, int_dev_labels, args.batch_size, use_cuda, False, True)
        # test_batches = reader.create_padded_batches(test_sent_berts, int_test_labels, args.batch_size, use_cuda, False, True)

        train_batches = [[x.to('cpu') for x in y] for y in pickle.load(open(os.path.join(args.save_path, "bert_train_batches_{}.pkl".format(args.suffix)), "rb"))]
        dev_batches = [[x.to('cpu') for x in y] for y in pickle.load(open(os.path.join(args.save_path, "bert_dev_batches_{}.pkl".format(args.suffix)), "rb"))]
        test_batches = [[x.to('cpu') for x in y] for y in pickle.load(open(os.path.join(args.save_path, "bert_test_batches_{}.pkl".format(args.suffix)), "rb"))]
        # suffix = "timebank" if "timebank" in args.data_dir else "litbank"
        # train_batches = [[x.to('cpu') for x in y] for y in pickle.load(open("bert_wlabel_train_batches_{}.pkl".format(suffix), "rb"))]
        # dev_batches = [[x.to('cpu') for x in y] for y in pickle.load(open("bert_wlabel_dev_batches_{}.pkl".format(suffix), "rb"))]
        # test_batches = [[x.to('cpu') for x in y] for y in pickle.load(open("bert_test_batches_{}.pkl".format(suffix), "rb"))]

    if args.model == 'word':
        model = EventExtractor(len(list(sent_vocab.keys())), args.emb_size, args.hidden_size, 1, args.dropout, args.bidir, args.model)
    elif args.model == 'delex':
        model = EventExtractor(len(list(pos_vocab.keys())), args.emb_size, args.hidden_size, 1, args.dropout, args.bidir, args.model)
    elif args.model == 'pos':
        model = EventExtractor(len(list(sent_vocab.keys())), args.emb_size, args.hidden_size, 1, args.dropout, args.bidir, args.model, pos_vocab_size=len(list(pos_vocab.keys())))
    elif args.model == 'bert-bilstm':
        print('Embedding size: {}'.format(train_batches[0][0].size()[-1]))
        model = EventExtractor(21541, train_batches[0][0].size()[-1], args.hidden_size, 1, args.dropout, args.bidir, args.model)
    elif args.model == 'bert-mlp':
        model = EventExtractor(10000, train_batches[0][0].size()[-1], args.hidden_size, 1, args.dropout, args.bidir, args.model, args.num_layers)

    if args.emb_file is not None:
        model.rep_learner.load_embeddings(args.emb_file, sent_vocab)
    if use_cuda:
        model = model.cuda()
    if args.do_train:
        train(model, train_batches, dev_batches, args.num_epochs, args.learning_rate, use_cuda, args.model_path+"_{}.pth".format(args.seed), args.model)
        if args.do_eval:
            test(model, test_batches, use_cuda, args.model_path+"_{}.pth".format(args.seed), args.model, oov=oov_vocab)
    else:
        test(model, test_batches, use_cuda, args.model_path+"_{}.pth".format(args.seed), args.model, oov=oov_vocab)
