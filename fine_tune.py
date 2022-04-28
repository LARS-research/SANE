import os
from datetime import datetime
import time
import argparse
import json
import pickle
import logging
import numpy as np

import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK
from logging_util import init_logger
from train4tune import main

sane_space ={'model': 'SANE',
         'hidden_size': hp.choice('hidden_size', [16, 32, 64, 128, 256]),
         'learning_rate': hp.uniform("lr", -3, -1.5),
         'weight_decay': hp.uniform("wr", -5, -3),
         'optimizer': hp.choice('opt', ['adagrad', 'adam']),
         'in_dropout': hp.choice('in_dropout', [0, 1, 2, 3, 4, 5, 6]),
         'out_dropout': hp.choice('out_dropout', [0, 1, 2, 3, 4, 5, 6]),
         'activation': hp.choice('act', ['relu', 'elu'])
         }

def get_args():
    parser = argparse.ArgumentParser("sane")
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--arch_filename', type=str, default='', help='given the location of searched res')
    parser.add_argument('--arch', type=str, default='', help='given the specific of searched res')
    parser.add_argument('--num_layers', type=int, default=3, help='num of GNN layers in SANE')
    parser.add_argument('--tune_topK', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--record_time', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--transductive', action='store_true', help='use transductive settings in train_search.')
    parser.add_argument('--with_linear', action='store_true', default=False, help='whether to use linear in NaOp')
    parser.add_argument('--with_layernorm', action='store_true', default=False, help='whether to use layer norm')
    parser.add_argument('--hyper_epoch', type=int, default=50, help='epoch in hyperopt.')
    parser.add_argument('--epochs', type=int, default=400, help='epoch in train GNNs.')
    parser.add_argument('--cos_lr', action='store_true', default=False, help='using lr decay in training GNNs.')
    parser.add_argument('--fix_last', type=bool, default=False, help='fix last layer in design architectures.')

    global args1
    args1 = parser.parse_args()

class ARGS(object):

    def __init__(self):
        super(ARGS, self).__init__()

def generate_args(arg_map):
    args = ARGS()
    for k, v in arg_map.items():
        setattr(args, k, v)
    for k, v in args1.__dict__.items():
        setattr(args, k, v)
    setattr(args, 'rnd_num', 1)

    args.learning_rate = 10**args.learning_rate
    args.weight_decay = 10**args.weight_decay
    args.in_dropout = args.in_dropout / 10.0
    args.out_dropout = args.out_dropout / 10.0
    args.save = '{}_{}'.format(args.data, datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
    args1.save = 'logs/tune-{}'.format(args.save)
    args.seed = 2
    args.grad_clip = 5
    args.momentum = 0.9
    return args

def objective(args):
    args = generate_args(args)
    vali_acc, test_acc, args = main(args)
    return {
        'loss': -vali_acc,
        'test_acc': test_acc,
        'status': STATUS_OK,
        'eval_time': round(time.time(), 2),
        }

def run_fine_tune():

    tune_str = time.strftime('%Y%m%d-%H%M%S')
    path = 'logs/tune-%s_%s' % (args1.data, tune_str)
    if not os.path.exists(path):
      os.mkdir(path)
    log_filename = os.path.join(path, 'log.txt')
    init_logger('fine-tune', log_filename, logging.INFO, False)

    lines = open(args1.arch_filename, 'r').readlines()

    suffix = args1.arch_filename.split('_')[-1][:-4] # need to re-write the suffix?

    test_res = []
    arch_set = set()
    if args1.data in ['small_Reddit', 'PubMed']:
        sane_space['hidden_size'] = hp.choice('hidden_size', [16, 32, 64])
    if args1.data == 'PPI':
        sane_space['learning_rate'] = hp.uniform("lr", -3, -1.6)
        sane_space['in_dropout'] = hp.choice('in_dropout', [0, 1])
        sane_space['out_dropout'] = hp.choice('out_dropout', [0, 1])
        sane_space['hidden_size'] = hp.choice('hidden_size', [64, 128, 256, 512, 1024])
    if args1.data == 'CiteSeer':
        sane_space['learning_rate'] = hp.uniform("lr", -2.5, -1.6)
        sane_space['weight_decay'] = hp.choice('wr', [-8])
        sane_space['in_dropout'] = hp.choice('in_dropout', [5])
        sane_space['out_dropout'] = hp.choice('out_dropout', [0])

    for ind, l in enumerate(lines):
        try:
            print('**********process {}-th/{}, logfilename={}**************'.format(ind+1, len(lines), log_filename))
            logging.info('**********process {}-th/{}**************'.format(ind+1, len(lines)))
            res = {}
            #iterate each searched architecture
            parts = l.strip().split(',')
            arch = parts[1].split('=')[1]
            args1.arch = arch
            if arch in arch_set:
                logging.info('the %s-th arch %s already searched....info=%s', ind+1, arch, l.strip())
                continue
            else:
                arch_set.add(arch)
            res['searched_info'] = l.strip()

            start = time.time()
            trials = Trials()
            #tune with validation acc, and report the test accuracy with the best validation acc
            best = fmin(objective, sane_space, algo=partial(tpe.suggest, n_startup_jobs=int(args1.hyper_epoch/5)),
                        max_evals=args1.hyper_epoch, trials=trials)

            space = hyperopt.space_eval(sane_space, best)
            print('best space is ', space)
            res['best_space'] = space
            args = generate_args(space)
            print('best args from space is ', args.__dict__)
            res['tuned_args'] = args.__dict__

            record_time_res = []
            c_vali_acc, c_test_acc = 0, 0
            #report the test acc with the best vali acc
            for d in trials.results:
                if -d['loss'] > c_vali_acc:
                    c_vali_acc = -d['loss']
                    c_test_acc = d['test_acc']
                    record_time_res.append('%s,%s,%s' % (d['eval_time'] - start, c_vali_acc, c_test_acc))
            res['test_acc'] = c_test_acc
            print('test_acc={}'.format(c_test_acc))
            #print('test_res=', res)

            test_accs=[]
            for i in range(5):
                vali_acc, t_acc, test_args = main(args)
                print('cal std: times:{}, valid_Acc:{}, test_acc:{}'.format(i,vali_acc,t_acc))
                test_accs.append(t_acc)
            test_accs = np.array(test_accs)
            print('test_results_5_times:{:.04f}+-{:.04f}'.format(np.mean(test_accs), np.std(test_accs)))
            test_res.append(res)

            test_res.append(res)
            with open('tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix), 'wb+') as fw:
                pickle.dump(test_res, fw)
            logging.info('**********finish {}-th/{}**************8'.format(ind+1, len(lines)))
        except Exception as e:
            logging.info('errror occured for %s-th, arch_info=%s, error=%s', ind+1, l.strip(), e)
            import traceback
            traceback.print_exc()
    print('finsh tunining {} archs, saved in {}'.format(len(arch_set), 'tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix)))


if __name__ == '__main__':
    get_args()
    if args1.arch_filename:
        run_fine_tune()


