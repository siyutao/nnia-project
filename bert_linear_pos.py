# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import time, random, functools
from torchtext.legacy import data
from torchtext.legacy import datasets
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from transformers import BertModel, BertTokenizer
import argparse

# check cuda / cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('DEVICE:', device)
# fix random seed for reproducability
SEED = 6456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# initial, padding and unknown tokens
init_token = tokenizer.cls_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

# print(init_token, pad_token, unk_token)

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

# print(init_token_idx, pad_token_idx, unk_token_idx)

max_input_length = tokenizer.max_model_input_sizes['bert-base-cased']


def cut_and_convert_to_id(tokens, tokenizer, max_input_length):
    tokens = tokens[:max_input_length - 1]
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    return tokens


def cut_to_max_length(tokens, max_input_length):
    tokens = tokens[:max_input_length - 1]
    return tokens


# preprocessors
text_preprocessor = functools.partial(cut_and_convert_to_id,
                                      tokenizer=tokenizer,
                                      max_input_length=max_input_length)

tag_preprocessor = functools.partial(cut_to_max_length,
                                     max_input_length=max_input_length)


def read_data(corpus_file, datafields):
    with open(corpus_file, encoding='utf-8') as f:
        examples = []
        words = []
        labels = []
        for line in f:
            line = line.strip()
            if line == '*':
                examples.append(data.Example.fromlist([words, labels], datafields))
                words = []
                labels = []
            else:
                columns = line.split()
                words.append(columns[1])
                labels.append(columns[2])
        return data.Dataset(examples, datafields)


class BERTPoSTagger(nn.Module):
    def __init__(self, bert, output_dim, dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]
        text = text.permute(1, 0)
        # text = [batch size, sent len]
        embedded = self.dropout(self.bert(text)[0])
        # embedded = [batch size, seq len, emb dim]
        embedded = embedded.permute(1, 0, 2)
        # embedded = [sent len, batch size, emb dim]
        predictions = self.fc(self.dropout(embedded))
        # predictions = [sent len, batch size, output dim]
        return predictions


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device)


def train(model, iterator, optimizer, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        text = batch.text
        tags = batch.label

        optimizer.zero_grad()
        # text = [sent len, batch size]
        predictions = model(text)
        # predictions = [sent len, batch size, output dim]
        # tags = [sent len, batch size]
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        # predictions = [sent len * batch size, output dim]
        # tags = [sent len * batch size]
        loss = criterion(predictions, tags)
        acc = categorical_accuracy(predictions, tags, tag_pad_idx)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            tags = batch.label
            predictions = model(text)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            loss = criterion(predictions, tags)
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def tag_sentence(model, device, sentence, tokenizer, text_field, tag_field):
    model.eval()

    if isinstance(sentence, str):
        tokens = tokenizer.tokenize(sentence)
    else:
        tokens = sentence

    numericalized_tokens = tokenizer.convert_tokens_to_ids(tokens)
    numericalized_tokens = [text_field.init_token] + numericalized_tokens

    unk_idx = text_field.unk_token

    unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]

    token_tensor = torch.LongTensor(numericalized_tokens)

    token_tensor = token_tensor.unsqueeze(-1).to(device)

    predictions = model(token_tensor)

    top_predictions = predictions.argmax(-1)

    predicted_tags = [tag_field.vocab.itos[t.item()] for t in top_predictions]

    predicted_tags = predicted_tags[1:]

    assert len(tokens) == len(predicted_tags)

    return tokens, predicted_tags, unks


def decode(model, device, tokenizer, text_field, tag_field):
    from sklearn import metrics
    gold, system = [], []
    model.eval()
    for item in test_data.examples:
        numericalized_tokens = [101] + item.text

        token_tensor = torch.LongTensor(numericalized_tokens)

        token_tensor = token_tensor.unsqueeze(-1).to(device)
        # print(token_tensor.shape)

        predictions = model(token_tensor)

        top_predictions = predictions.argmax(-1)

        predicted_tags = [tag_field.vocab.itos[t.item()] for t in top_predictions][1:]
        assert len(predicted_tags) == len(item.label)
        system.extend(predicted_tags)
        gold.extend(item.label)

    report = metrics.classification_report(gold, system, digits=4)
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--epochs', type=int, default=2, help='num of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout rate')
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    N_EPOCHS = args.epochs
    DROPOUT = args.dropout

    TEXT = data.Field(use_vocab=False, lower=False, preprocessing=text_preprocessor,
                      init_token=init_token_idx, pad_token=pad_token_idx, unk_token=unk_token_idx)

    LABEL = data.Field(unk_token=None, init_token='<pad>', preprocessing=tag_preprocessor)

    # corresponding fields in the data
    fields = [('text', TEXT), ('label', LABEL)]
    # load data from the tsv files (already split)
    train_data = read_data('data/ontonotes_splits/train.tsv', fields)
    valid_data = read_data('data/ontonotes_splits/dev.tsv', fields)
    test_data = read_data('data/ontonotes_splits/test.tsv', fields)
    print(f"Size of training set  : {len(train_data)}")
    print(f"Size of validation set: {len(valid_data)}")
    print(f"Size of testing set   : {len(test_data)}")

    LABEL.build_vocab(train_data)

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), sort=False, batch_size=BATCH_SIZE, device=device)

    bert = BertModel.from_pretrained('bert-base-cased')
    OUTPUT_DIM = len(LABEL.vocab)
    model = BERTPoSTagger(bert, OUTPUT_DIM, DROPOUT)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    TAG_PAD_IDX = LABEL.vocab.stoi[LABEL.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)
    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'linear-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    # evaluate
    model.load_state_dict(torch.load('linear-model.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion, TAG_PAD_IDX)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    # inference
    sentence = 'The Queen will deliver a speech about the conflict in North Korea at 1pm tomorrow.'
    tokens, tags, unks = tag_sentence(model, device, sentence, tokenizer, TEXT, LABEL)
    print("Pred. Tag\tToken\n")
    for token, tag in zip(tokens, tags):
        print(f"{tag}\t\t{token}")

    # decode
    decode(model, device, tokenizer, TEXT, LABEL)
