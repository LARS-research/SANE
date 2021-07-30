import os
import os.path as osp
import sys
import glob
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
import torch.nn.functional as F

from torch.autograd import Variable
from model import NetworkGNN as Network
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import add_self_loops
from logging_util import init_logger

parser = argparse.ArgumentParser("Cora")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--hidden_size', type=int, default=64, help='embedding size in NN')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#log_format = '%(asctime)s %(message)s'
#logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#    format=log_format, datefmt='%m/%d %I:%M:%S %p')

#CIFAR_CLASSES = 10

def generate_K_edge_index(edge_index, K_list):
  index_list = edge_index.tolist()
  nbr_map = {}
  for i in range(len(index_list[0])):
    a, b = index_list[0][i], index_list[1][i]
    nbr_map.setdefault(a, []).append(b)
  res = {}
  for K in K_list:
    K_index = []
    for k, v in sorted(nbr_map.items(), key=lambda d: d[0]):
      n = K if K < len(v) else len(v)
      samples = np.random.choice(v, n)
      K_index.extend([(k, r) for r in samples])
    K_index = torch.tensor(K_index).transpose(0, 1)
    res[str(K)] = K_index
  return res

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def gen_uniform_60_20_20_split(data):
  skf = StratifiedKFold(5, shuffle=True)
  idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]
  return cat(idx[:3], 0), cat(idx[3:4], 0), cat(idx[4:], 0)

def save_load_split(dataset, run, gen_splits):
    data = dataset[0]
    prefix = gen_splits.__name__[4:-6]
    path = osp.join(dataset.raw_dir, '..', '{}_{:03d}.pt'.format(prefix, run))

    if osp.exists(path):
        split = torch.load(path)
    else:
        split = gen_splits(data)
        torch.save(split, path)

    data.train_mask = index_to_mask(split[0], data.num_nodes)
    data.val_mask = index_to_mask(split[1], data.num_nodes)
    data.test_mask = index_to_mask(split[2], data.num_nodes)

    return data

def main(test_args):
  if not test_args is None:
    args = test_args

  args.save = 'logs/eval-{}'.format(args.save)
  log_filename = os.path.join(args.save, 'log_test_res.txt')
  init_logger('', log_filename, logging.INFO, False)
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  #path = osp.join('../data', 'Cora')
  if args.data == 'Amazon_Computers':
    dataset = Amazon('../data/Amazon_Computers', 'Computers')
  elif args.data == 'Coauthor_Physics':
    dataset = Coauthor('../data/Coauthor_Physics', 'Physics')
  elif args.data == 'Coauthor_CS':
    dataset = Coauthor('../data/Coauthor_CS', 'CS')
  elif args.data == 'Cora_Full':
    dataset = CoraFull('../data/Cora_Full')
  data = save_load_split(dataset, 1, gen_uniform_60_20_20_split)
  edge_index, _ =  add_self_loops(data.edge_index)
  data.edge_index = edge_index
  #data.edge_index_map = generate_K_edge_index(data.edge_index, [5, 10, 15, 20, 25])
  hidden_size = args.hidden_size

  genotype = args.arch
  logging.info('genotype = %s', genotype)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  model = Network(genotype, criterion, dataset.num_features, dataset.num_classes, hidden_size)
  model = model.cuda()
  utils.load(model, args.model_path)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  #model.drop_path_prob = args.drop_path_prob
  test_acc, test_obj = infer(data, model, criterion)
  logging.info('test_acc %f', test_acc)
  return test_acc, args.save

def infer(data, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    logits = F.log_softmax(model(data.to(device)), dim=-1)

  #for step, (input, target) in enumerate(test_queue):
  #input = Variable(input, volatile=True).cuda()
  #target = Variable(target, volatile=True).cuda(async=True)
  input = logits[data.test_mask].to(device)
  target = data.y[data.test_mask].to(device)

  #logits, _ = model(input)
  loss = criterion(input, target)

  pred = logits[data.test_mask].max(1)[1]
  acc = (pred == target).sum().item() / data.test_mask.sum().item()
  logging.info('test acc=%s', acc)

  prec1, prec5 = utils.accuracy(input, target, topk=(1, 5))
  n = input.size(0)
  objs.update(loss.data.item(), n)
  top1.update(prec1.data.item(), n)
  top5.update(prec5.data.item(), n)

  #if step % args.report_freq == 0:
  logging.info('test %e %f %f', objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

if __name__ == '__main__':
  main()

