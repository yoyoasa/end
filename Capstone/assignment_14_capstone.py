from attention_model import Encoder, Decoder, Seq2Seq
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from spacy.tokenizer import Tokenizer
import re
import json
import pandas as pd
import time
import math
import random
import numpy as np
import spacy
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from torchtext.legacy.data import Field, Example, BucketIterator, Dataset
import torchtext
import torch.optim as optim
import torch.nn as nn
import torch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, "0" to  "7"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print('Device:', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# SEED = 1234

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True


def custom_tokenizer(nlp):
    infix_re = re.compile(
        r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'\(\)\[\]\{\}\*\%\^\+\-\=\<\>\|\!(//)(\n)(\t)~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                     suffix_search=suffix_re.search,
                     infix_finditer=infix_re.finditer,
                     token_match=None)


spacy_que = spacy.load('en_core_web_sm')
spacy_ans = spacy.load('en_core_web_sm')
spacy_ans.tokenizer = custom_tokenizer(spacy_ans)


def tokenize_que(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_que.tokenizer(text)]


def tokenize_ans(text):
    """
    Tokenizes Code text from a string into a list of strings
    """
    return [tok.text for tok in spacy_ans.tokenizer(text)]


Question = Field(init_token='<sos>', eos_token='<eos>',
                 batch_first=True, tokenize=tokenize_que, lower=True)
Answer = Field(init_token='<sos>', eos_token='<eos>',
               batch_first=True, tokenize=tokenize_ans, lower=True)

fields = [('questions', Question), ('answers', Answer)]

train_lines = []
data = {"questions": "", "answers": ""}

max_length = 150
max_decoder_length = max_length+50

with open('./english_python_data.txt', 'r') as f:
    for l in f:
        if l.startswith('# write'):
            data['questions'] = data['questions'].rstrip('\n')
            data['questions'] = data['questions'].lstrip('#')
            data['questions'] = data['questions'].lstrip()
            data['questions'] = data['questions'].rstrip()
            data['answers'] = data['answers'].rstrip('\n')
            if len(tokenize_ans(data['answers'])) < max_length-1:
                train_lines.append(data)
            data = {"questions": l, "answers": ""}
        else:
            if not l == '\n':
                data['answers'] = data['answers'] + l

with open('./conala-test.json', 'r') as f:
    records = json.load(f)
    for record in records:
        data['questions'] = record['intent']
        data['answers'] = record['snippet']
        train_lines.append(data)

with open('./conala-train.json', 'r') as f:
    records = json.load(f)
    for record in records:
        data['questions'] = record['intent']
        data['answers'] = record['snippet']
        train_lines.append(data)

# train_lines.pop()

random.shuffle(train_lines)

dataset_length = len(train_lines)

print("Data Set Size", dataset_length)

valid_size = int(dataset_length * 0.1)

print("Valid Data Size", valid_size)


## As we have already proved that this network works well we will use all data to train our network
train_data = pd.DataFrame(train_lines)
valid_data = pd.DataFrame(train_lines[dataset_length-valid_size:])

train_data = [Example.fromlist(
    [train_data.questions[i], train_data.answers[i]], fields) for i in range(train_data.shape[0])]
valid_data = [Example.fromlist(
    [valid_data.questions[i], valid_data.answers[i]], fields) for i in range(valid_data.shape[0])]

train_data = Dataset(train_data, fields)
valid_data = Dataset(valid_data, fields)

Question.build_vocab(train_data, min_freq=2)
Answer.build_vocab(train_data, vectors=torchtext.vocab.Vectors(
    "./python_code_glove_embedding_300.txt"), min_freq=2)

print(f"Unique tokens in Question vocabulary: {len(Question.vocab)}")
print(f"Unique tokens in Answer vocabulary: {len(Answer.vocab)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32

print('BATCH_SIZE:', 32)

train_iterator, valid_iterator = BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE,
    sort=False,
    device=device)


INPUT_DIM = len(Question.vocab)
OUTPUT_DIM = len(Answer.vocab)
HID_DIM = 300
ENC_LAYERS = 4
DEC_LAYERS = 4
ENC_HEADS = 5
DEC_HEADS = 5
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device,
              max_decoder_length)

SRC_PAD_IDX = Question.vocab.stoi[Question.pad_token]
TRG_PAD_IDX = Answer.vocab.stoi[Answer.pad_token]

print('INPUT_DIM:', INPUT_DIM)
print('OUTPUT_DIM:', OUTPUT_DIM)
print('HID_DIM:', HID_DIM)
print('ENC_LAYERS:', ENC_LAYERS)
print('DEC_LAYERS:', DEC_LAYERS)
print('ENC_HEADS:', ENC_HEADS)
print('DEC_HEADS:', DEC_HEADS)
print('ENC_PF_DIM:', ENC_PF_DIM)
print('DEC_PF_DIM:', DEC_PF_DIM)
print('ENC_DROPOUT:', ENC_DROPOUT)
print('DEC_DROPOUT:', DEC_DROPOUT)
print('SRC_PAD_IDX:', SRC_PAD_IDX)
print('TRG_PAD_IDX:', TRG_PAD_IDX)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model.apply(initialize_weights)

# Apply pretrained weights
glove_pretrained_embeddings = Answer.vocab.vectors
print("Pretrained embedding dimension:", glove_pretrained_embeddings.shape)
model.decoder.tok_embedding.weight.data = glove_pretrained_embeddings.to(
    device)

LEARNING_RATE = 0.0005
MAX_LR = 0.001
N_EPOCHS = 24
CLIP = 1
STEPS_PER_EPOCH = len(train_iterator)

print('LEARNING_RATE:', LEARNING_RATE)
print('MAX_LR:', MAX_LR)
print('N_EPOCHS:', N_EPOCHS)
print('CLIP:', CLIP)
print('STEPS_PER_EPOCH:', STEPS_PER_EPOCH)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# One Cycle Scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=MAX_LR, steps_per_epoch=STEPS_PER_EPOCH, epochs=N_EPOCHS, anneal_strategy='linear')

# # One cycle schedule with custome function
# schedule = np.interp(np.arange(N_EPOCHS+1), [0, 5, 20, N_EPOCHS], [LEARNING_RATE, MAX_LR, LEARNING_RATE/5.0, LEARNING_RATE/10.0])
# def lr_schedules(epoch):
#     return schedule[epoch+1]

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src = batch.questions
        trg = batch.answers

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch.questions
            trg = batch.answers

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_valid_loss = float('inf')
best_file = 'capstone-model.pt'

for epoch in range(N_EPOCHS):

    # optimizer.param_groups[0]['lr'] = lr_schedules(epoch)

    print('\nLearning Rate:', optimizer.param_groups[0]["lr"])

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_file = 'capstone-model-{}-{}.pt'.format(epoch, best_valid_loss)
        torch.save(model.state_dict(), best_file)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(
        f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(
        f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

print("Best Model:", best_file)
model.load_state_dict(torch.load(best_file))

# test_loss = evaluate(model, valid_iterator, criterion)

# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):

    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('en_core_web_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(
                trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention


def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):

    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):

        ax = fig.add_subplot(n_rows, n_cols, i+1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'],
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


escapes = ['.', ',', '?', ':', ';', '...', '‘', '’', '`', '“', '”', '"', '\'',
           '(', ')', '[', ']', '{', '}', '*', '%', '^', '\n', '\t', '~', '=', '_', '-', '+', '>', '<', '!', '/']


def joiner(txt):
    prev = txt[0]
    code = prev
    for current in txt[1:]:
        if current == '<eos>':
            return code
        if prev not in escapes and current not in escapes:
            code = code + " " + current
        else:
            code = code + "" + current
        prev = current
    return code


def generate_code(idx):
    question = vars(train_data.examples[idx])['questions']
    answer = vars(train_data.examples[idx])['answers']

    translation, attention = translate_sentence(
        question, Question, Answer, model, device, max_decoder_length)

    print("####", joiner(question), "####")
    print("------------------------------------------------------------")
    print(joiner(answer))
    print("------------------------------------------------------------")
    print(joiner(translation))
    print("------------------------------------------------------------")


# generate_code(1)
# generate_code(252)
# generate_code(21)
# generate_code(786)
# generate_code(151)
# generate_code(553)
# generate_code(1001)

while (1):
    val = input("Enter your question or type 'exit': ")
    if val == 'exit':
        break
    else:
        question = tokenize_que(val)

        translation, attention = translate_sentence(
            question, Question, Answer, model, device, max_decoder_length)

        print("####", joiner(question), "####")
        print("------------------------------------------------------------")
        print(joiner(translation))
        print("------------------------------------------------------------")
