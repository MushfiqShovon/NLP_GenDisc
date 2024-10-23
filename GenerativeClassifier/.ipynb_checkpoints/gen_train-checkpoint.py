#!/usr/bin/python

# python gen.py --cuda --dataset dbpedia --datafile traindata.v40000.l80.s5 --epochs 10

import sys
import os
import time
import random
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils_train import resume_checkpoint, save_checkpoint, parse_args
from models import Gen

def batches(data, label, batch_size, is_eval = False, nclass = 0):
    d_l = list(zip(data, label))
    random.shuffle(d_l)
    data, label = zip(*d_l)
    data, label = list(data), list(label)

    for i in range(0, len(data), batch_size):
        sentences = data[i:i + batch_size]
        labels = label[i:i + batch_size]

        s_l = zip(sentences, labels)
        s_l = sorted(s_l, key = lambda l: len(l[0]), reverse=True)

        sentences, labels = zip(*s_l)

        sentences = list(sentences)
        labels = list(labels)

        # x_pred: predicted ground truth, padding in the end
        # y_ext: extend label to the length of sentence length for concatnation
        # seq_len: pred_seq_len = actual seq len - 1

        y_ext = []
        for idx, d in enumerate(sentences):
            y_ext.append([labels[idx]] * (len(d) - 1))

        if is_eval:
            y_exts = []
            for y_label in range(nclass):
                y_ext = []
                for d in sentences:
                    y_ext.append(torch.LongTensor([y_label] * (len(d) - 1)))
                y_exts.append(y_ext)

            yield [torch.LongTensor(s) for s in sentences], y_exts, torch.LongTensor(labels)
        else:
            yield [torch.LongTensor(s) for s in sentences], \
                [torch.LongTensor(y) for y in y_ext], torch.LongTensor(labels)

def evaluate(validdata, validlabel, model, criterion, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    total_correct = 0.

    cnt = 0
    with torch.no_grad():
        for sents, y_exts, labels in batches(validdata, validlabel, args.batch_size, True, args.nclass):
            hidden = model.init_hidden(len(sents))

            x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
            x_pred = nn.utils.rnn.pack_sequence([s[1:] for s in sents])

            # p_y = torch.FloatTensor([0.071] * len(seq_len))

            losses = []
            for y_ext in y_exts:
                y_ext = nn.utils.rnn.pack_sequence(y_ext)

                if args.device.type == 'cuda':
                    x, y_ext, x_pred, labels = x.cuda(), y_ext.cuda(), x_pred.cuda(), labels.cuda()

                # output (batch_size, )
                hidden = model.init_hidden(len(sents))
                loss = model(x, x_pred, y_ext, hidden, criterion, True)
                losses.append(loss)

            losses = torch.cat(losses, dim=0).view(-1, len(sents))
            prediction = torch.argmin(losses, dim=0)

            num_correct = (prediction == labels).float().sum()

            total_loss += torch.sum(torch.min(losses, dim=0)[0]).item()
            total_correct += num_correct.item()
            cnt += 1

    return total_loss / cnt, total_correct / len(validlabel) * 100.0

def train_epoch(traindata, trainlabel, validdata, validlabel, model, criterion, optimizer, args, epoch):
    # https://discuss.pytorch.org/t/how-to-properly-use-hidden-states-for-rnn/13607
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch_ind, sents_label in enumerate(batches(traindata, trainlabel, args.batch_size)):
        '''
            sents: a batch of sentences with padding
            x_pred: ground truth of the prediction of the LM
            seq_len: (the length of sentence - 1) in the batch before padding
            y_ext: extended labels in the batch, for concat operation
            labels: the ground truth label
        '''
        sents, y_ext, labels = sents_label

        batch_ind += 1
        optimizer.zero_grad()
        hidden = model.init_hidden(len(sents))

        x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
        x_pred = nn.utils.rnn.pack_sequence([s[1:] for s in sents])
        y_ext = nn.utils.rnn.pack_sequence(y_ext)

        # p_y = torch.FloatTensor([0.071] * len(seq_len))

        if args.device.type == 'cuda':
            x, y_ext, x_pred = x.cuda(), y_ext.cuda(), x_pred.cuda() 

        loss = model(x, x_pred, y_ext, hidden, criterion)

        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # num_correct = (torch.max(output, 1)[1].view(labels.size()).data == labels.data).float().sum()
        # acc = 100.0 * num_correct / labels.size(0)

        total_loss += loss.item()

        if batch_ind % args.log_interval == 0:
            elapsed = time.time() - start_time
            _, train_acc = evaluate(sents, labels, model, criterion, args)
            val_loss, val_acc = evaluate(validdata, validlabel, model, criterion, args)

            logging.info('| epoch %d | %5d/%5d batches | lr %5.5f | ms/batch %5.2f | '
                    'train loss %5.2f | train acc %5.2f | valid loss %.2f | valid acc %.2f',
                epoch, batch_ind, len(trainlabel) // args.batch_size, args.lr,
                elapsed * 1000 / args.log_interval, loss.item(), train_acc,
                val_loss, val_acc)
            model.train()

    return total_loss / batch_ind #, total_correct / len(trainlabel)

def save_model(epoch, model, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, f'ModelParameter_{epoch}.pth')
    torch.save(model.state_dict(), filename)
    logging.info(f"Model saved to {filename}")


def main(args=sys.argv[1:]):
    args = parse_args(args)
    args.nclass = len(open(os.path.join('data', args.dataset, 'classes.txt'), 'r').readlines())
    
    logging.basicConfig(level=args.logging)
    logging.info('Args: %s', args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            logging.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # load data
    data_dict = torch.load(os.path.join('data', args.dataset, 'data', args.datafile))
    traindata = data_dict['traindata']
    trainlabel = data_dict['trainlabel']
    validdata = data_dict['validdata']
    validlabel = data_dict['validlabel']
    testdata = data_dict['testdata']
    testlabel = data_dict['testlabel']
    args.vocab_size = data_dict['vocabsize']

    model = Gen(args.vocab_size, args.word_emb_dim, args.label_emb_dim, args.hid_dim, \
        args.nlayers, args.nclass, args.dropout, args.cuda, args.tied, args.use_bias, \
        args.concat_label, args.avg_loss, args.one_hot)
    criterion = nn.CrossEntropyLoss(reduce=False)

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] 
        for x in params if x.size())
    logging.info('Model total parameters: %d', total_params)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

    args.start_epoch = 0
    best_val_acc = None

    if args.resume:
        args, model, optimizer = resume_checkpoint(args, model, optimizer, logging)

    try:
        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()
            logging.info("Training epoch %d", epoch)

            train_loss = train_epoch(traindata, trainlabel, validdata, validlabel, model, 
                criterion, optimizer, args, epoch)

            val_loss, val_acc = evaluate(validdata, validlabel, model, criterion, args)

            logging.info('-' * 89)
            logging.info('| end of epoch %3d | time: %.2f s | train loss %5.2f | valid loss %.2f | '
                    'valid acc %.2f', epoch, (time.time() - epoch_start_time), train_loss, 
                                               val_loss, val_acc)
            logging.info('-' * 89)

            is_best = not best_val_acc or val_acc > best_val_acc

            # Save the model if the validation loss is the best we've seen so far.

            if (epoch+1) % args.save_interval == 0:
                save_checkpoint({
                    'args': args,
                    'state_dict': model.state_dict(),
                    'best_val_acc': best_val_acc,
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                }, args.dataset, args.save_prefix + '_' + str(args.save_interval) + '.chkp')
            if is_best:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                save_checkpoint({
                    'args': args,
                    'state_dict': model.state_dict(),
                    'best_val_acc': best_val_acc,
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                }, args.dataset, args.save_prefix + '_best.chkp')
    except KeyboardInterrupt:
        logging.info('-' * 89)
        logging.info('Exiting from training early')

    checkpoint = torch.load(os.path.join('experiment', args.dataset, args.save_prefix + '_best.chkp'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model_dir=os.path.join('experiment', args.dataset, 'pth')
    
    save_model(args.epochs, model, model_dir)

    model.eval()
    test_loss, test_acc = evaluate(testdata, testlabel, model, criterion, args)
    logging.info('=' * 89)
    logging.info('| Test | test loss %.2f | test acc %.2f', test_loss, test_acc)
    logging.info('=' * 89)

if __name__ == '__main__':
    main()