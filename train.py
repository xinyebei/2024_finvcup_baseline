# -*- coding: utf-8 -*-

import argparse, glob, os, torch, warnings, time
from utils.tools import *
from utils.dataLoader import train_loader
from utils.FakeModel import FakeModel

parser = argparse.ArgumentParser()
parser.add_argument('--device',      type=str,   default='cuda:0',       help='Device training on ')
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=32,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="./data/finvcup9th_1st_ds4/finvcup9th_1st_ds4_train_data.csv",     help='The path of the training list, eg:"/data08/VoxCeleb2/train_list.txt" in my case')
parser.add_argument('--eval_list',  type=str,   default="./data/finvcup9th_1st_ds4/finvcup9th_1st_ds4_valid_data.csv",              help='The path of the evaluation list, eg:"/data08/VoxCeleb1/veri_test2.txt" in my case')
parser.add_argument('--save_path',  type=str,   default="./exps",                                     help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')

## Model and Loss settings
parser.add_argument('--n_class', type=int,   default=2,   help='Number of class')

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

## Define the data loader
trainloader = train_loader(**vars(args))
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

## Search for the exist models
#modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
#modelfiles.sort()
modelfiles = []

## Only do evaluation, the initial_model is necessary
if args.eval == True:
    s = FakeModel(**vars(args))
    print("Model %s loaded from previous state!"%args.initial_model)
    s.load_parameters(args.initial_model)
    EER, minDCF = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
    print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
    quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
    print("Model %s loaded from previous state!"%args.initial_model)
    s = FakeModel(**vars(args))
    s.load_parameters(args.initial_model)
    epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
    print("Model %s loaded from previous state!"%modelfiles[-1])
    epoch = 1
    s = FakeModel(**vars(args))
    s.load_parameters(modelfiles[-1])
    ## Otherwise, system will train from scratch
else:
    epoch = 1
    s = FakeModel(**vars(args))

ACCs = []
RECALLs = []
F1s = []
score_file = open(args.score_save_path, "a+")

while(1):
    ## Training for one epoch
    loss, lr, acc, recall, f1 = s.train_network(epoch = epoch, loader = trainLoader)

    ## Evaluation every [test_step] epochs
    if epoch % args.test_step == 0:
        s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
        acc_pred, recall_pred, f1_pred = s.eval_network(eval_list = args.eval_list, num_frames=args.num_frames)
        
        ACCs.append(acc_pred)
        RECALLs.append(recall_pred)
        F1s.append(f1_pred)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, Acc %2.2f%%, Recall %2.2f%%, F1 %2.2f%%, Val_acc %2.2f%%, Best_acc %2.2f%%, Val_recall %2.2f%%, Best_recall %2.2f%%, Val_f1  %2.2f%%, Best_f1 %2.2f%%"%(epoch, acc, recall, f1, ACCs[-1], max(ACCs), RECALLs[-1], max(RECALLs), F1s[-1], max(F1s)))
        score_file.write("%d epoch, Acc %2.2f%%, Recall %2.2f%%, F1 %2.2f%%, Val_acc %2.2f%%, Best_acc %2.2f%%, Val_recall %2.2f%%, Best_recall %2.2f%%, Val_f1  %2.2f%%, Best_f1 %2.2f%%\n"%(epoch, acc, recall, f1, ACCs[-1], max(ACCs), RECALLs[-1], max(RECALLs), F1s[-1], max(F1s)))
        score_file.flush()

    if epoch >= args.max_epoch:
        quit()

    epoch += 1
