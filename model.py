import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path
from torch_geometric.nn import LayerNorm

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

class NaOp(nn.Module):
  def __init__(self, primitive, in_dim, out_dim, act, with_linear=False):
    super(NaOp, self).__init__()

    self._op = NA_OPS[primitive](in_dim, out_dim)
    self.op_linear = nn.Linear(in_dim, out_dim)
    self.act = act_map(act)
    self.with_linear = with_linear

  def forward(self, x, edge_index):
    if self.with_linear:
      return self.act(self._op(x, edge_index)+self.op_linear(x))
    else:
      return self.act(self._op(x, edge_index))
# class NaMLPOp(nn.Module):
#     def __init__(self, primitive, in_dim, out_dim, act):
#         super(NaMLPOp, self).__init__()
#         self._op = NA_MLP_OPS[primitive](in_dim, out_dim)
#         self.act = act_map(act)
# 
#     def forward(self, x, edge_index):
#         return self.act(self._op(x, edge_index))

class ScOp(nn.Module):
    def __init__(self, primitive):
        super(ScOp, self).__init__()
        self._op = SC_OPS[primitive]()

    def forward(self, x):
        return self._op(x)

class LaOp(nn.Module):
    def __init__(self, primitive, hidden_size, act, num_layers=None):
        super(LaOp, self).__init__()
        self._op = LA_OPS[primitive](hidden_size, num_layers)
        self.act = act_map(act)

    def forward(self, x):
        return self.act(self._op(x))

class NetworkGNN(nn.Module):
    '''
        implement this for sane.
        Actually, sane can be seen as the combination of three cells, node aggregator, skip connection, and layer aggregator
        for sane, we dont need cell, since the DAG is the whole search space, and what we need to do is implement the DAG.
    '''
    def __init__(self, genotype, criterion, in_dim, out_dim, hidden_size, num_layers=3, in_dropout=0.5, out_dropout=0.5, act='relu', is_mlp=False, args=None):
        super(NetworkGNN, self).__init__()
        self.genotype = genotype
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.in_dropout = in_dropout
        self.out_dropout = out_dropout
        self._criterion = criterion
        ops = genotype.split('||')
        self.args = args

        #node aggregator op
        self.lin1 = nn.Linear(in_dim, hidden_size)

        self.gnn_layers = nn.ModuleList(
                [NaOp(ops[i], hidden_size, hidden_size, act, with_linear=args.with_linear) for i in range(num_layers)])

        #skip op
        if self.args.fix_last:
            if self.num_layers > 1:
                self.sc_layers = nn.ModuleList([ScOp(ops[i+num_layers]) for i in range(num_layers - 1)])
            else:
                self.sc_layers = nn.ModuleList([ScOp(ops[num_layers])])
        else:
            # no output conditions.
            skip_op = ops[num_layers:2 * num_layers]
            if skip_op == ['none'] * num_layers:
                skip_op[-1] = 'skip'
                print('skip_op:', skip_op)
            self.sc_layers = nn.ModuleList([ScOp(skip_op[i]) for i in range(num_layers)])

        #layer norm
        self.lns = torch.nn.ModuleList()
        if self.args.with_layernorm:
            for i in range(num_layers):
                self.lns.append(LayerNorm(hidden_size, affine=True))

        #layer aggregator op
        self.layer6 = LaOp(ops[-1], hidden_size, 'linear', num_layers)

        # self.classifier = nn.Linear(hidden_size, out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim))
        #self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        #generate weights by softmax
        x = self.lin1(x)
        x = F.dropout(x, p=self.in_dropout, training=self.training)
        js = []
        for i in range(self.num_layers):
            x = self.gnn_layers[i](x, edge_index)
            if self.args.with_layernorm:
                # layer_norm = nn.LayerNorm(normalized_shape=x.size(), elementwise_affine=False)
                # x = layer_norm(x)
                x = self.lns[i](x)
            x = F.dropout(x, p=self.in_dropout, training=self.training)
            if i == self.num_layers - 1 and self.args.fix_last:
                js.append(x)
            else:
                js.append(self.sc_layers[i](x))
        x5 = self.layer6(js)
        x5 = F.dropout(x5, p=self.out_dropout, training=self.training)

        logits = self.classifier(x5)
        return logits

    def _loss(self, logits, target):
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        num_na_ops = len(NA_PRIMITIVES)
        num_sc_ops = len(SC_PRIMITIVES)
        num_la_ops = len(LA_PRIMITIVES)

        #self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.na_alphas = Variable(1e-3*torch.randn(self.num_layers, num_na_ops).cuda(), requires_grad=True)
        if self.num_layers > 1:
            self.sc_alphas = Variable(1e-3*torch.randn(self.num_layers - 1, num_sc_ops).cuda(), requires_grad=True)
        else:
            self.sc_alphas = Variable(1e-3*torch.randn(1, num_sc_ops).cuda(), requires_grad=True)
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


