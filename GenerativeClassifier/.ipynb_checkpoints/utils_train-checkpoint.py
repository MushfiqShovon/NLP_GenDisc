import torch
import os
from argparse import ArgumentParser

def resume_checkpoint(args, model, optimizer, logging):
    resume_file_name = os.path.join('experiment', args.dataset, args.resume)
    if os.path.isfile(resume_file_name):
        logging.info("=> loading checkpoint '{}'".format(resume_file_name))
        checkpoint = torch.load(resume_file_name)
        args.start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer']) 
    else:
        logging.info("=> no checkpoint found at '{}'".format(args.resume))
    return args, model, optimizer

def save_checkpoint(state, dataset, filename):
    save_dir = os.path.join('experiment', dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, filename))

def parse_args(args):
    argp = ArgumentParser()

    # model related
    argp.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, GRU)')
    argp.add_argument('--mode', type=str, default='auxiliary',
                    help='type of latent model (auxiliary, joint, middle, hierarchical)')
    argp.add_argument("--word_emb_dim", type=int, default=100,
                    help="Word embedding dimensionality")
    argp.add_argument("--label_emb_dim", type=int, default=100,
                    help="label embedding dimensionality")
    argp.add_argument('--cond_emb_dim', type=int, default=100,
                    help='condition variable embedding dimensionality')
    argp.add_argument('--hid_dim', type=int, default=100,
                    help='number of hidden units per layer')
    argp.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
    argp.add_argument('--ncondition', type=int, default=30,
                    help='number of condition variable')
    argp.add_argument('--use_bias', action="store_true",
                    help='use a different bias for each label')
    argp.add_argument('--concat_label', type=str, default='hidden',
                    help='type of concat (hidden, input, both)')
    argp.add_argument('--avg_loss', action="store_true")
    argp.add_argument("--one_hot", action="store_true",
                    help='use one hot embedding')
    argp.add_argument("--bit_width", type=int, default=8,
                    help="Bitwidth for quantization")

    # training related
    argp.add_argument('--dataset', type=str,  default='dbpedia',
                    help='dataset name')
    argp.add_argument('--datafile', type=str,  default='traindata.v40000.l80.s5',
                    help='datafile name')
    argp.add_argument('--optimizer', type=str,  default='adam',
                    help='optimizer to use (sgd, adam)')
    argp.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
    argp.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
    argp.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
    argp.add_argument("--use_EM", action="store_true")
    argp.add_argument("--bilinear", action="store_true")
    argp.add_argument("--tied", action="store_true")
    argp.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
    argp.add_argument("--epochs", type=int, default=100)
    argp.add_argument("--batch_size", type=int, default=32)

    # inference related
    argp.add_argument('--infer_method', type=str, default='sum',
                    help='(sum, posterior, max)')

    # interpret related
    argp.add_argument('--idx2wordfile', type=str,  default='data/train_sample',
                    help='idx2wordfile file')
    argp.add_argument('--validfile', type=str,  default='data/train_sample',
                    help='idx2wordfile file') 

    # logging and saving related
    argp.add_argument("--logging", choices=["INFO", "DEBUG"],
                      default="INFO")
    argp.add_argument('--log_interval', type=int, default=1000, metavar='N',
                    help='report interval')
    argp.add_argument('--save_prefix', type=str, default='gen',
                    help='path to save the final model')
    argp.add_argument('--save_interval', type=int, default=10, metavar='N',
                    help='save interval (epoch)')
    argp.add_argument('--resume', type=str, default=None,
                    help='path to save the final model')

    # misc
    argp.add_argument('--seed', type=int, default=0,
                    help='random seed')
    argp.add_argument("--cuda", action="store_true")

    return argp.parse_args(args)