import os
import os.path as osp
import sys
import time
import glob
import pickle
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch import cat
from sklearn.metrics import f1_score

from torch.autograd import Variable
from model import NetworkGNN as Network
from utils import gen_uniform_60_20_20_split, save_load_split
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, Reddit, PPI
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import add_self_loops
from logging_util import init_logger
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

def main(exp_args):
    global train_args
    train_args = exp_args

    tune_str = time.strftime('%Y%m%d-%H%M%S')
    train_args.save = 'logs/tune-{}-{}'.format(train_args.data, tune_str)
    if not os.path.exists(train_args.save):
        os.mkdir(train_args.save)

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    #np.random.seed(train_args.seed)
    torch.cuda.set_device(train_args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(train_args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(train_args.seed)

    if train_args.data == 'Amazon_Computers':
        dataset = Amazon('../data/Amazon_Computers', 'Computers')
    elif train_args.data == 'Coauthor_Physics':
        dataset = Coauthor('../data/Coauthor_Physics', 'Physics')
    elif train_args.data == 'Coauthor_CS':
        dataset = Coauthor('../data/Coauthor_CS', 'CS')
    elif train_args.data == 'Cora_Full':
        dataset = CoraFull('../data/Cora_Full')
    elif train_args.data == 'PubMed':
        dataset = Planetoid('../data/', 'PubMed')
    elif train_args.data == 'Cora':
        dataset = Planetoid('../data/', 'Cora')
    elif train_args.data == 'CiteSeer':
        dataset = Planetoid('../data/', 'CiteSeer')
    elif train_args.data == 'PPI':
        train_dataset = PPI('../data/PPI', split='train')
        val_dataset = PPI('../data/PPI', split='val')
        test_dataset = PPI('../data/PPI', split='test')
        ppi_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        ppi_val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        ppi_test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        # print('load PPI done!')
        data = [ppi_train_loader, ppi_val_loader, ppi_test_loader]

    if train_args.data == 'small_Reddit':
        dataset = Reddit('../data/Reddit/')
        with open('../data/small_Reddit/sampled_reddit.obj', 'rb') as f:
            data = pickle.load(f)
            raw_dir = '../data/small_Reddit/raw/'
    elif train_args.data == 'small_arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        with open('../data/small_arxiv/sampled_arxiv.obj', 'rb') as f:
            data = pickle.load(f)
            raw_dir = '../data/small_arxiv/raw/'
    genotype = train_args.arch
    hidden_size = train_args.hidden_size

    if train_args.data != 'PPI':
        raw_dir = dataset.raw_dir
        data = dataset[0]
        data = save_load_split(data, raw_dir, 1, gen_uniform_60_20_20_split)

        edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
        data.edge_index = edge_index
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
        num_features = train_dataset.num_features
        num_classes = train_dataset.num_classes
    criterion = criterion.cuda()
    model = Network(genotype, criterion, num_features, num_classes, hidden_size,
                    num_layers=train_args.num_layers, in_dropout=train_args.in_dropout,
                    out_dropout=train_args.out_dropout, act=train_args.activation,
                    is_mlp=False, args=train_args)
    model = model.cuda()

    logging.info("genotype=%s, param size = %fMB, args=%s", genotype, utils.count_parameters_in_MB(model), train_args.__dict__)

    if train_args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            train_args.learning_rate,
            #momentum=args.momentum,
            weight_decay=train_args.weight_decay
            )
    elif train_args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            train_args.learning_rate,
            momentum=train_args.momentum,
            weight_decay=train_args.weight_decay
            )
    elif train_args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            train_args.learning_rate,
            weight_decay=train_args.weight_decay
            )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs))

    val_res = 0
    best_val_acc = best_test_acc = 0
    for epoch in range(train_args.epochs):
        train_acc, train_obj = train(train_args.data, data, model, criterion, optimizer)
        if train_args.cos_lr:
            scheduler.step()

        valid_acc, valid_obj = infer(train_args.data, data, model, criterion)
        test_acc, test_obj = infer(train_args.data, data, model, criterion, test=True)

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_test_acc = test_acc

        if epoch % 10 == 0:
            logging.info('epoch=%s, lr=%s, train_obj=%s, train_acc=%f, valid_acc=%s, test_acc=%s', epoch, scheduler.get_lr()[0], train_obj, train_acc, best_val_acc, best_test_acc)

        utils.save(model, os.path.join(train_args.save, 'weights.pt'))

    return best_val_acc, best_test_acc, train_args

def train(dataset_name, data, model, criterion, optimizer):
    if dataset_name == 'PPI':
        return train_ppi(data, model, criterion, optimizer)
    else:
        return train_trans(data, model, criterion, optimizer)

def infer(dataset_name, data, model, criterion, test=False):
    if dataset_name == 'PPI':
        return infer_ppi(data, model, criterion, test=test)
    else:
        return infer_trans(data, model, criterion, test=test)

def train_trans(data, model, criterion, optimizer):

    mask = data.train_mask
    model.train()
    target = data.y[mask].to(device)

    optimizer.zero_grad()
    logits = model(data.to(device))

    input = logits[mask].to(device)

    loss = criterion(input, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)
    optimizer.step()

    acc = logits[mask].max(1)[1].eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc, loss/mask.sum().item()



def infer_trans(data, model, criterion, test=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        logits = model(data.to(device))
    if test:
        mask = data.test_mask
    else:
        mask = data.val_mask
    input = logits[mask].to(device)
    target = data.y[mask].to(device)
    loss = criterion(input, target)

    acc = logits[mask].max(1)[1].eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc, loss/mask.sum().item()

    # prec1, prec5 = utils.accuracy(input, target, topk=(1, 3))
    # n = data.val_mask.sum().item()
    # objs.update(loss.data.item(), n)
    # top1.update(prec1.data.item(), n)
    # top5.update(prec5.data.item(), n)
    # return top1.avg, objs.avg

def train_ppi(data, model, criterion, optimizer):
    model.train()
    preds, ys = [], []
    total_loss = 0
    # input all data

    for train_data in data[0]:
        train_data = train_data.to(device)
        target = Variable(train_data.y).to(device)

        # train loss
        optimizer.zero_grad()
        input = model(train_data).to(device)
        loss = criterion(input, target)
        total_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)
        optimizer.step()

        preds.append((input > 0).float().cpu())
        ys.append(train_data.y.cpu())
    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    prec1 = f1_score(y, pred, average='micro')
    # print('train_loss:', total_loss / len(data[0].dataset))
    return prec1, total_loss / len(data[0].dataset)

def infer_ppi(data, model, criterion, test=False):
    model.eval()
    total_loss = 0
    preds, ys = [], []
    if test:
        infer_data = data[2]
    else:
        infer_data = data[1]

    for val_data in infer_data:
        val_data = val_data.to(device)
        with torch.no_grad():
            logits = model(val_data).to(device)

        loss = criterion(logits, val_data.y.to(device))
        total_loss += loss.item()

        preds.append((logits > 0).float().cpu())
        ys.append(val_data.y.cpu())
    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    prec1 = f1_score(y, pred, average='micro')
    return prec1, total_loss / len(infer_data.dataset)

if __name__ == '__main__':
  main()



