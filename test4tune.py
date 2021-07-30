import os
import os.path as osp
import sys
import glob
import numpy as np
import torch
import utils
import logging
import pickle
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import cat

from torch.autograd import Variable
from model import NetworkGNN as Network
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, Reddit
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import add_self_loops
from logging_util import init_logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def gen_uniform_60_20_20_split(data):
  skf = StratifiedKFold(5, shuffle=True)
  idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]
  return cat(idx[:3], 0), cat(idx[3:4], 0), cat(idx[4:], 0)

def save_load_split(data, raw_dir, run, gen_splits):
  prefix = gen_splits.__name__[4:-6]
  path = osp.join(raw_dir, '..', '{}_{:03d}.pt'.format(prefix, run))

  if osp.exists(path):
      split = torch.load(path)
  else:
      split = gen_splits(data)
      torch.save(split, path)

  data.train_mask = index_to_mask(split[0], data.num_nodes)
  data.val_mask = index_to_mask(split[1], data.num_nodes)
  data.test_mask = index_to_mask(split[2], data.num_nodes)

  return data

def main(test_args1):
  global test_args
  test_args = test_args1

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  #np.random.seed(test_args.seed)
  torch.cuda.set_device(test_args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(test_args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(test_args.seed)

  #path = osp.join('../data', 'Cora')
  if test_args.data == 'Amazon_Computers':
    dataset = Amazon('../data/Amazon_Computers', 'Computers')
  elif test_args.data == 'Coauthor_Physics':
    dataset = Coauthor('../data/Coauthor_Physics', 'Physics')
  elif test_args.data == 'Coauthor_CS':
    dataset = Coauthor('../data/Coauthor_CS', 'CS')
  elif test_args.data == 'Cora_Full':
    dataset = CoraFull('../data/Cora_Full')
  elif test_args.data == 'PubMed':
    dataset = Planetoid('../data/PubMed', 'PubMed')
  elif test_args.data == 'Cora':
    dataset = Planetoid('../data/Cora', 'Cora')
  elif test_args.data == 'CiteSeer':
    dataset = Planetoid('../data/CiteSeer', 'CiteSeer')

  if test_args.data == 'small_Reddit':
    dataset = Reddit('../data/Reddit/')
    with open('../data/small_Reddit/sampled_reddit.obj', 'rb') as f:
      data = pickle.load(f)
      raw_dir = '../data/small_Reddit/raw/'
  else:
      raw_dir = dataset.raw_dir
      data = dataset[0]

  data = save_load_split(data, raw_dir, test_args.rnd_num, gen_uniform_60_20_20_split)
  edge_index, _ =  add_self_loops(data.edge_index)
  data.edge_index = edge_index
  hidden_size = test_args.hidden_size

  genotype = test_args.arch

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  model = Network(genotype, criterion, dataset.num_features, dataset.num_classes, hidden_size, num_layers=test_args.num_layers, in_dropout=test_args.in_dropout, out_dropout=test_args.out_dropout, act=test_args.activation)
  model = model.cuda()
  utils.load(model, test_args.model_path)

  logging.info("gpu=%s, genotype=%s, param size = %fMB, args=%s", test_args.gpu,  genotype, utils.count_parameters_in_MB(model), test_args.__dict__)

  test_acc, test_obj = infer(data, model, criterion)
  logging.info('test_acc=%f', test_acc)
  return test_acc, test_args.save

def infer(data, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    logits = F.log_softmax(model(data.to(device)), dim=-1)

  input = logits[data.test_mask].to(device)
  target = data.y[data.test_mask].to(device)

  #logits, _ = model(input)
  loss = criterion(input, target)

  pred = logits[data.test_mask].max(1)[1]
  acc = (pred == target).sum().item() / data.test_mask.sum().item()

  prec1, prec5 = utils.accuracy(input, target, topk=(1, 3))
  n = input.size(0)
  objs.update(loss.data.item(), n)
  top1.update(prec1.data.item(), n)
  top5.update(prec5.data.item(), n)

  return top1.avg, objs.avg

if __name__ == '__main__':
  main()
