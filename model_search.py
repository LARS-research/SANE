import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operations import *
from torch.autograd import Variable
from genotypes import NA_PRIMITIVES,  SC_PRIMITIVES, LA_PRIMITIVES

def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")

class NaMixedOp(nn.Module):

  def __init__(self, in_dim, out_dim, with_linear):
    super(NaMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.with_linear = with_linear

    for primitive in NA_PRIMITIVES:
      op = NA_OPS[primitive](in_dim, out_dim)
      self._ops.append(op)

      if with_linear:
        self._ops_linear = nn.ModuleList()
        op_linear = torch.nn.Linear(in_dim, out_dim)
        self._ops_linear.append(op_linear)

  def forward(self, x, weights, edge_index, ):
    mixed_res = []
    if self.with_linear:
      for w, op, linear in zip(weights, self._ops, self._ops_linear):
        mixed_res.append(w * F.elu(op(x, edge_index)+linear(x)))
    else:
      for w, op in zip(weights, self._ops):
        mixed_res.append(w * F.elu(op(x, edge_index)))
    return sum(mixed_res)

class ScMixedOp(nn.Module):

  def __init__(self):
    super(ScMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in SC_PRIMITIVES:
      op = SC_OPS[primitive]()
      self._ops.append(op)

  def forward(self, x, weights):
    mixed_res = []
    for w, op in zip(weights, self._ops):
      mixed_res.append(w * op(x))
    return sum(mixed_res)

class LaMixedOp(nn.Module):

  def __init__(self, hidden_size, num_layers=None):
    super(LaMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in LA_PRIMITIVES:
      op = LA_OPS[primitive](hidden_size, num_layers)
      self._ops.append(op)

  def forward(self, x, weights):
    mixed_res = []
    for w, op in zip(weights, self._ops):
      mixed_res.append(w * F.relu(op(x)))
    return sum(mixed_res)

class Network(nn.Module):
  '''
      implement this for sane.
      Actually, sane can be seen as the combination of three cells, node aggregator, skip connection, and layer aggregator
      for sane, we dont need cell, since the DAG is the whole search space, and what we need to do is implement the DAG.
  '''

  def __init__(self, criterion, in_dim, out_dim, hidden_size, num_layers=3, dropout=0.5, epsilon=0.0, with_conv_linear=False, args=None):
    super(Network, self).__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self._criterion = criterion
    self.dropout=dropout
    self.epsilon = epsilon
    self.explore_num = 0
    self.with_linear = with_conv_linear
    self.args = args

    #node aggregator op
    self.lin1 = nn.Linear(in_dim, hidden_size)
    self.layer1 = NaMixedOp(hidden_size, hidden_size,self.with_linear)
    self.layer2 = NaMixedOp(hidden_size, hidden_size,self.with_linear)
    self.layer3 = NaMixedOp(hidden_size, hidden_size,self.with_linear)

    #skip op
    self.layer4 = ScMixedOp()
    self.layer5 = ScMixedOp()
    if not self.args.fix_last:
        self.layer6 = ScMixedOp()

    #layer aggregator op
    self.layer7 = LaMixedOp(hidden_size, num_layers)

    self.classifier = nn.Linear(hidden_size, out_dim)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._criterion, self.in_dim, self.out_dim, self.hidden_size).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, data, discrete=False):
    x, edge_index = data.x, data.edge_index
    #prob = float(np.random.choice(range(1,11), 1) / 10.0)
    
    self.na_weights = F.softmax(self.na_alphas, dim=-1)
    self.sc_weights = F.softmax(self.sc_alphas, dim=-1)
    self.la_weights = F.softmax(self.la_alphas, dim=-1)

    #generate weights by softmax
    x = self.lin1(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    x1 = self.layer1(x, self.na_weights[0], edge_index)
    x1 = F.dropout(x1, p=self.dropout, training=self.training)
    x2 = self.layer2(x1, self.na_weights[1], edge_index)
    x2 = F.dropout(x2, p=self.dropout, training=self.training)
    x3 = self.layer3(x2, self.na_weights[2], edge_index)
    x3 = F.dropout(x3, p=self.dropout, training=self.training)

    if self.args.fix_last:
        x4 = (x3, self.layer4(x1, self.sc_weights[0]), self.layer5(x2, self.sc_weights[1]))
    else:
        x4 = (self.layer4(x1, self.sc_weights[0]), self.layer5(x2, self.sc_weights[1]), self.layer6(x3, self.sc_weights[2]))

    x5 = self.layer7(x4, self.la_weights[0])
    x5 = F.dropout(x5, p=self.dropout, training=self.training)

    logits = self.classifier(x5)
    return logits

  def _loss(self, data, is_valid=True):
      logits = self(data)
      if is_valid:
          input = logits[data.val_mask].cuda()
          target = data.y[data.val_mask].cuda()
      else:
          input = logits[data.train_mask].cuda()
          target = data.y[data.train_mask].cuda()
      return self._criterion(input, target)

  def _loss_ppi(self, data, is_valid=True):
      input = self(data).cuda()
      target = data.y.cuda()
      return self._criterion(input, target)

  def _initialize_alphas(self):
    #k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_na_ops = len(NA_PRIMITIVES)
    num_sc_ops = len(SC_PRIMITIVES)
    num_la_ops = len(LA_PRIMITIVES)

    #self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.na_alphas = Variable(1e-3*torch.randn(3, num_na_ops).cuda(), requires_grad=True)
    if self.args.fix_last:
        self.sc_alphas = Variable(1e-3*torch.randn(2, num_sc_ops).cuda(), requires_grad=True)
    else:
        self.sc_alphas = Variable(1e-3*torch.randn(3, num_sc_ops).cuda(), requires_grad=True)

    self.la_alphas = Variable(1e-3*torch.randn(1, num_la_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.na_alphas,
      self.sc_alphas,
      self.la_alphas,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(na_weights, sc_weights, la_weights):
      gene = []
      na_indices = torch.argmax(na_weights, dim=-1)
      for k in na_indices:
          gene.append(NA_PRIMITIVES[k])
      #sc_indices = sc_weights.argmax(dim=-1)
      sc_indices = torch.argmax(sc_weights, dim=-1)
      for k in sc_indices:
          gene.append(SC_PRIMITIVES[k])
      #la_indices = la_weights.argmax(dim=-1)
      la_indices = torch.argmax(la_weights, dim=-1)
      for k in la_indices:
          gene.append(LA_PRIMITIVES[k])
      return '||'.join(gene)

    gene = _parse(F.softmax(self.na_alphas, dim=-1).data.cpu(), F.softmax(self.sc_alphas, dim=-1).data.cpu(), F.softmax(self.la_alphas, dim=-1).data.cpu())


    return gene

  def sample_arch(self):

    num_na_ops = len(NA_PRIMITIVES)
    num_sc_ops = len(SC_PRIMITIVES)
    num_la_ops = len(LA_PRIMITIVES)

    gene = []
    for i in range(3):
        op = np.random.choice(NA_PRIMITIVES, 1)[0]
        gene.append(op)
    for i in range(2):
        op = np.random.choice(SC_PRIMITIVES, 1)[0]
        gene.append(op)
    op = np.random.choice(LA_PRIMITIVES, 1)[0]
    gene.append(op)
    return '||'.join(gene)

  def get_weights_from_arch(self, arch):
    arch_ops = arch.split('||')
    #print('arch=%s' % arch)
    num_na_ops = len(NA_PRIMITIVES)
    num_sc_ops = len(SC_PRIMITIVES)
    num_la_ops = len(LA_PRIMITIVES)


    na_alphas = Variable(torch.zeros(3, num_na_ops).cuda(), requires_grad=True)
    sc_alphas = Variable(torch.zeros(2, num_sc_ops).cuda(), requires_grad=True)
    la_alphas = Variable(torch.zeros(1, num_la_ops).cuda(), requires_grad=True)

    for i in range(3):
        ind = NA_PRIMITIVES.index(arch_ops[i])
        na_alphas[i][ind] = 1

    for i in range(3, 5):
        ind = SC_PRIMITIVES.index(arch_ops[i])
        sc_alphas[i-3][ind] = 1

    ind = LA_PRIMITIVES.index(arch_ops[5])
    la_alphas[0][ind] = 1

    arch_parameters = [na_alphas, sc_alphas, la_alphas]
    return arch_parameters

  def set_model_weights(self, weights):
    self.na_weights = weights[0]
    self.sc_weights = weights[1]
    self.la_weights = weights[2]
    #self._arch_parameters = [self.na_alphas, self.sc_alphas, self.la_alphas]


