# -*- coding: utf-8 -*-

import os, numpy, torch
from sklearn import metrics
from operator import itemgetter
import torch.nn.functional as F

def init_args(args):
    args.score_save_path    = os.path.join(args.save_path, 'score.txt')
    args.model_save_path    = os.path.join(args.save_path, 'model')
    os.makedirs(args.model_save_path, exist_ok = True)
    return args

def metrics_scores(output, target):
    output = output.detach().cpu().numpy().argmax(axis=1)
    target = target.detach().cpu().numpy()
    
    accuracy = metrics.accuracy_score(target, output)
    recall = metrics.recall_score(target, output)
    precision = metrics.precision_score(target, output)
    F1 = metrics.f1_score(target, output)   
    
    return accuracy*100, recall*100, precision, F1
    
    