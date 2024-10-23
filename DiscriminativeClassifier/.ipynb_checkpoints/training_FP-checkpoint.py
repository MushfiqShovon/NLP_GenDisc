import torch
import pandas as pd
import re
import spacy
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
import torchtext
print(torchtext.__version__)
import time
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Training script")

parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--bit_width', type=int, default=8, help='bit width for quantization')
parser.add_argument('--model', type=str, default='LSTMNet', choices=['LSTMNet', 'QLSTM', 'QLSTM_2bit', 'QLSTM_1bit'], help='model class to use')

args = parser.parse_args()

# Set the number of epochs from command line arguments
EPOCH_NUMBER = args.epochs

from models import LSTMNet, QLSTM, QLSTM_2bit, QLSTM_1bit 

import site
print(site.getsitepackages())
import os
os.environ['SP_DIR'] = '/opt/conda/lib/python3.11/site-packages'

accuracies = []

# nlp library of Pytorch
from torchtext import data

import warnings as wrn
wrn.filterwarnings('ignore')


SEED = 2021

torch.manual_seed(SEED)
torch.backends.cuda.deterministic = True


def clean_text(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9]+', ' ', str(text))
    return cleaned_text

# Load and preprocess the data files
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, header=None, delimiter='\t') # Assuming tab-separated values in .data files
    df[1] = df[1].apply(clean_text) # Assuming the text is in the second column
    cleaned_file_path = file_path.replace('.data', '_cleaned.data')
    df.to_csv(cleaned_file_path, index=False, header=False)
    return cleaned_file_path

cleaned_train_file = load_and_preprocess('./data/train.data')
cleaned_valid_file = load_and_preprocess('./data/valid.data')
cleaned_test_file = load_and_preprocess('./data/test.data')


spacy_en = spacy.load('en_core_web_sm')

def spacy_tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

LABEL = data.LabelField()
TEXT = data.Field(tokenize=spacy_tokenizer, batch_first=True, include_lengths=True)
fields = [("label", LABEL), ("text", TEXT)]

training_data = data.TabularDataset(path=cleaned_train_file, format="csv", fields=fields, skip_header=True)
validation_data = data.TabularDataset(path=cleaned_valid_file, format="csv", fields=fields, skip_header=True)
test_data = data.TabularDataset(path=cleaned_test_file, format="csv", fields=fields, skip_header=True)

print(vars(training_data.examples[0]))

train_data,valid_data = training_data.split(split_ratio=0.75,
                                            random_state=random.seed(SEED))

TEXT.build_vocab(train_data,
                 min_freq=5)

LABEL.build_vocab(train_data)
# Count the number of instances per class
label_counts = {LABEL.vocab.itos[i]: LABEL.vocab.freqs[LABEL.vocab.itos[i]] for i in range(len(LABEL.vocab))}
print("Number of instances per class:", label_counts)


print("Size of text vocab:",len(TEXT.vocab))

print("Size of label vocab:",len(LABEL.vocab))

TEXT.vocab.freqs.most_common(10)

# Creating GPU variable
#device = torch.device("cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Sent_SIZE=32
print("Batch size initialized")

# We'll create iterators to get batches of data when we want to use them
"""
This BucketIterator batches the similar length of samples and reduces the need of 
padding tokens. This makes our future model more stable

"""
train_iterator,validation_iterator = data.BucketIterator.splits(
    (train_data,valid_data),
    batch_size = Sent_SIZE,
    # Sort key is how to sort the samples
    sort_key = lambda x:len(x.text),
    sort_within_batch = True,
    device = device
)

test_iterator = data.BucketIterator(
    test_data,
    batch_size=Sent_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device
)


SIZE_OF_VOCAB = len(TEXT.vocab)
EMBEDDING_DIM = 100
NUM_HIDDEN_NODES = 100
NUM_OUTPUT_NODES = len(LABEL.vocab)
NUM_LAYERS = 1
BIDIRECTION = False
DROPOUT = 0.2
BIT_WIDTH = args.bit_width

#Model creation and model directory for savings the checkpoint files and accuracy result

if args.model == 'LSTMNet':
    model = LSTMNet(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT, BIT_WIDTH)
    MODEL_DIR = 'ModelParameterDisc/ModelParameter_LSTM_FP'
elif args.model == 'QLSTM':
    model = QLSTM(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT, BIT_WIDTH)
    MODEL_DIR = f'ModelParameterDisc/ModelParameter_LSTM_{BIT_WIDTH}bit'
elif args.model == 'QLSTM_1bit':
    model = QLSTM_1bit(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT, BIT_WIDTH)
    MODEL_DIR = f'ModelParameterDisc/ModelParameter_LSTM_{BIT_WIDTH}bit'
elif args.model == 'QLSTM_2bit':
    model = QLSTM_2bit(SIZE_OF_VOCAB, EMBEDDING_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT, BIT_WIDTH)
    MODEL_DIR = f'ModelParameterDisc/ModelParameter_LSTM_{BIT_WIDTH}bit'

criterion = nn.CrossEntropyLoss()

print(torch.cuda.is_available())

model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-3)

model


def multi_class_accuracy(preds, y):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model,iterator,optimizer,criterion):
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    model.train()
    
    for batch in iterator:
        
        # cleaning the cache of optimizer
        optimizer.zero_grad()
        
        text,text_lengths = batch.text
        #print("Text Length:", text_lengths[0].item())
        global Sent_SIZE
        #Sent_SIZE=text_lengths[0].item()
        #print("Sent Length:", Sent_SIZE)
        #print("Iterator Batch Size:", batch.batch_size)
        batch.batch_size=Sent_SIZE
        #print("Iterator Batch Size:", batch.batch_size)
        iterator = data.BucketIterator(
            train_data,
            batch_size=Sent_SIZE,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            device=device
        )
        
        # forward propagation and squeezing
        predictions = model(text,text_lengths).squeeze()
        
        # computing loss / backward propagation
        loss = criterion(predictions, batch.label)
        #loss = criterion(predictions,batch.type)
        loss.backward()
        
        # accuracy
        acc = multi_class_accuracy(predictions,batch.label)
        
        # updating params
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    # It'll return the means of loss and accuracy
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model,iterator,criterion):
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    # deactivate the dropouts
    model.eval()
    
    # Sets require_grad flat False
    with torch.no_grad():
        for batch in iterator:
            text,text_lengths = batch.text
            
            predictions = model(text,text_lengths).squeeze()
              
            #compute loss and accuracy
            loss = criterion(predictions, batch.label)
            acc = multi_class_accuracy(predictions, batch.label)
            
            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def infer(model, iterator, criterion):
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze()
            loss = criterion(predictions, batch.label)
            acc = multi_class_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Create dataset and iterator for test data
test_fields = [("label", LABEL), ("text", TEXT)]
test_data = data.TabularDataset(path=cleaned_test_file, format="csv", fields=test_fields, skip_header=True)


# Create the directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("total number of epoch:",EPOCH_NUMBER)
for epoch in range(1,EPOCH_NUMBER+1):
    
    print("======================================================")
    print("Epoch: %d" %epoch)
    print("======================================================")
    
    start_time = time.time()
    
    train_loss,train_acc = train(model,train_iterator,optimizer,criterion)
    
    valid_loss,valid_acc = evaluate(model,validation_iterator,criterion)
    
    end_time = time.time()
    epoch_duration = end_time - start_time
    # Showing statistics
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    print(f'\tTime taken for epoch: {epoch_duration:.2f} seconds\n')
    print()
    
    epoch_save_path = f'./{MODEL_DIR}/model_epoch_{epoch}.pth'
    torch.save(model.state_dict(), epoch_save_path)
    print(f'Model parameters saved for epoch {epoch} at "{epoch_save_path}"')
    
    accuracies.append({'Epoch': epoch, 'Train Accuracy': train_acc, 'Validation Accuracy': valid_acc, 'Epoch Time': epoch_duration})

test_loss, test_acc = infer(model, test_iterator, criterion)
print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')

accuracies.append({'Epoch': 'Test', 'Train Accuracy': None, 'Validation Accuracy': None, 'Epoch Time': None, 'Test Accuracy': test_acc})
df_accuracies = pd.DataFrame(accuracies)
accuracy_save_path = f'./{MODEL_DIR}/accuracies.csv'
df_accuracies.to_csv(accuracy_save_path, index=False)
print(f'Accuracies saved at "{accuracy_save_path}"')

model_parameters = model.state_dict()
print(model_parameters)
torch.save(model.state_dict(), f'./{MODEL_DIR}/modelParameter.pth')