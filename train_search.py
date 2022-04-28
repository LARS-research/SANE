import os
import os.path as osp
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch import cat
import pickle
from sklearn.metrics import f1_score

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from utils import gen_uniform_60_20_20_split, save_load_split
from torch_geometric.data import DataLoader

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, Reddit, PPI
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import StratifiedKFold
from logging_util import init_logger

parser = argparse.ArgumentParser("sane-train-search")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--record_time', action='store_true', default=False, help='used for run_with_record_time func')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=4, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--epsilon', type=float, default=0.0, help='the explore rate in the gradient descent process')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--transductive', action='store_true', help='use transductive settings in train_search.')
parser.add_argument('--with_conv_linear', type=bool, default=False, help=' in NAMixOp with linear op')
parser.add_argument('--fix_last', type=bool, default=False, help='fix last layer in design architectures.')
parser.add_argument('--num_layers', type=int, default=3, help='num of aggregation layers')

args = parser.parse_args()

def main():
    global device
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    args.save = 'logs/search-{}'.format(args.save)
    if not os.path.exists(args.save):
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_filename = os.path.join(args.save, 'log.txt')
    init_logger('', log_filename, logging.INFO, False)
    print('*************log_filename=%s************' % log_filename)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args.__dict__)

    if args.data == 'Amazon_Computers':
        dataset = Amazon('../data/Amazon_Computers', 'Computers')
    elif args.data == 'Coauthor_Physics':
        dataset = Coauthor('../data/Coauthor_Physics', 'Physics')
    elif args.data == 'Coauthor_CS':
        dataset = Coauthor('../data/Coauthor_CS', 'CS')
    elif args.data == 'Cora_Full':
        dataset = CoraFull('../data/Cora_Full')
    elif args.data == 'PubMed':
        dataset = Planetoid('../data/', 'PubMed')
    elif args.data == 'Cora':
        dataset = Planetoid('../data/', 'Cora')
    elif args.data == 'CiteSeer':
        dataset = Planetoid('../data/', 'CiteSeer')
    elif args.data == 'PPI':
        train_dataset = PPI('../data/PPI', split='train')
        val_dataset = PPI('../data/PPI', split='val')
        test_dataset = PPI('../data/PPI', split='test')
        ppi_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        ppi_val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        ppi_test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        # print('load PPI done!')
        data = [ppi_train_loader, ppi_val_loader, ppi_test_loader]
    if args.data == 'small_Reddit':
        dataset = Reddit('../data/Reddit/')
        with open('../data/small_Reddit/sampled_reddit.obj', 'rb') as f:
            data = pickle.load(f)
            raw_dir = '../data/small_Reddit/raw/'
    elif args.data == 'small_arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        with open('../data/small_arxiv/sampled_arxiv.obj', 'rb') as f:
            data = pickle.load(f)
            raw_dir = '../data/small_arxiv/raw/'


    # if not args.transductive:
        #622 split
        # data = save_load_split(data, raw_dir, 1, gen_uniform_60_20_20_split)
    if args.data != 'PPI':
        raw_dir = dataset.raw_dir
        data = dataset[0]
        data = save_load_split(data, raw_dir, 1, gen_uniform_60_20_20_split)

        edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
        data.edge_index = edge_index
        hidden_size = 32

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        model = Network(criterion, dataset.num_features, dataset.num_classes, hidden_size, num_layers=args.num_layers, epsilon=args.epsilon, args=args)
    else:
        hidden_size = 16
        criterion = nn.BCEWithLogitsLoss()
        criterion = criterion.cuda()
        model = Network(criterion, train_dataset.num_features, train_dataset.num_classes, hidden_size, epsilon=args.epsilon, with_conv_linear=args.with_conv_linear, args=args)

    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)# send model to compute validation loss
    # test_acc_with_time = []
    # cur_t = 0
    search_cost = 0
    for epoch in range(args.epochs):
        t1 = time.time()
        lr = scheduler.get_lr()[0]
        if epoch % 1 == 0:
            logging.info('epoch %d lr %e', epoch, lr)
            genotype = model.genotype()
            logging.info('genotype = %s', genotype)


        train_acc, train_obj = train(args.data, data, model, architect, criterion, optimizer, lr)
        scheduler.step()
        t2 = time.time()
        search_cost += (t2 - t1)

        valid_acc, valid_obj = infer(args.data, data, model, criterion)
        test_acc,  test_obj = infer(args.data, data, model, criterion, test=True)

        if epoch % 1 == 0:
            logging.info('epoch=%s, train_acc=%f, valid_acc=%f, test_acc=%f, explore_num=%s', epoch, train_acc, valid_acc,test_acc, model.explore_num)
            print('epoch={}, train_acc={:.04f}, valid_acc={:.04f}, test_acc={:.04f},explore_num={}'.format(epoch, train_acc, valid_acc, test_acc, model.explore_num))
        utils.save(model, os.path.join(args.save, 'weights.pt'))
    logging.info('The search process costs %.2fs', search_cost)
    return genotype

def train(dataset_name, data, model, architect, criterion, optimizer, lr):
    if dataset_name =='PPI':
        return train_ppi(data, model, architect, criterion, optimizer, lr)
    else:
        return train_trans(data, model, architect, criterion, optimizer, lr)

def infer(dataset_name, data, model, criterion, test=False):
    if dataset_name == 'PPI':
        return infer_ppi(data, model, criterion, test=test)
    else:
        return infer_trans(data, model, criterion, test=test)

def train_trans(data, model, architect, criterion, optimizer, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.train()
    mask = data.train_mask
    target = Variable(data.y[mask], requires_grad=False).to(device)


    #architecture send input or send logits, which are important for computation in architecture
    architect.step(data.to(device), lr, optimizer, unrolled=args.unrolled)

    #train loss
    logits = model(data.to(device))
    input = logits[mask].to(device)

    optimizer.zero_grad()
    loss = criterion(input, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    acc = logits[mask].max(1)[1].eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc, loss/mask.sum().item()
    # prec1, prec5 = utils.accuracy(input, target, topk=(1, 3))
    # n = input.size(0)
    # objs.update(loss.data.item(), n)
    # top1.update(prec1.data.item(), n)
    # top5.update(prec5.data.item(), n)
    #
    # return top1.avg, objs.avg

def infer_trans(data, model, criterion, test=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        logits = model(data.to(device))
    if test:
        mask = data.test_mask
        # input = logits[].to(device)
        # target = data.y[data.test_mask].to(device)
        # loss = criterion(input, target)
        # print('test_loss:', loss.item())
    else:
        mask = data.val_mask
        # input = logits[data.val_mask].to(device)
        # target = data.y[data.val_mask].to(device)
        # loss = criterion(input, target)
        # print('valid_loss:', loss.item())
    input = logits[mask].to(device)
    target = data.y[mask].to(device)
    loss = criterion(input, target)
    acc = input.max(1)[1].eq(target).sum().item() / mask.sum().item()
    return acc, loss/mask.sum().item()
    # prec1, prec5 = utils.accuracy(input, target, topk=(1, 3))
    # n = data.val_mask.sum().item()
    # objs.update(loss.data.item(), n)
    # top1.update(prec1.data.item(), n)
    # top5.update(prec5.data.item(), n)

    # return top1.avg, objs.avg
def train_ppi(data, model, architect, criterion, optimizer, lr):
    model.train()
    total_loss = 0

    preds, ys = [], []
    total_loss = 0
    # input all data
    architect.step(data, lr, optimizer, unrolled=args.unrolled)

    for train_data in data[0]:
        train_data = train_data.to(device)
        target = Variable(train_data.y).to(device)
        # train loss
        optimizer.zero_grad()
        input = model(train_data).to(device)
        loss = criterion(input, target)
        total_loss += loss.item()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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

def run_by_seed():
    res = []
    for i in range(5):
        print('searched {}-th for {}...'.format(i+1, args.data))
        args.save = '{}-{}'.format(args.data, time.strftime("%Y%m%d-%H%M%S"))
        seed = np.random.randint(0, 10000)
        args.seed = seed
        genotype = main()
        res.append('seed={},genotype={},saved_dir={}'.format(seed, genotype, args.save))
    filename = 'exp_res/%s-searched_res-%s-eps%s-reg%s.txt' % (args.data, time.strftime('%Y%m%d-%H%M%S'), args.epsilon, args.weight_decay)
    fw = open(filename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print('searched res for {} saved in {}'.format(args.data, filename))


if __name__ == '__main__':
    run_by_seed()


