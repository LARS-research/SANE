import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, JumpingKnowledge
from torch_geometric.nn import GINConv
from pyg_gnn_layer import GeoLayer
from gin_conv import GINConv2
from gcn_conv import GCNConv2
from geniepath import GeniePathLayer

# from genotypes import NA_MLP_PRIMITIVES

NA_OPS = {
  'sage': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sage'),
  'sage_sum': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sum'),
  'sage_max': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'max'),
  'gcn': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gcn'),
  'gat': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gat'),
  'gin': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gin'),
  'gat_sym': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gat_sym'),
  'gat_linear': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'linear'),
  'gat_cos': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'cos'),
  'gat_generalized_linear': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'generalized_linear'),
  'geniepath': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'geniepath'),
}


SC_OPS={
  'none': lambda: Zero(),
  'skip': lambda: Identity(),
  }

LA_OPS={
  'l_max': lambda hidden_size, num_layers: LaAggregator('max', hidden_size, num_layers),
  'l_concat': lambda hidden_size, num_layers: LaAggregator('cat', hidden_size, num_layers),
  'l_lstm': lambda hidden_size, num_layers: LaAggregator('lstm', hidden_size, num_layers)
}

class NaAggregator(nn.Module):

  def __init__(self, in_dim, out_dim, aggregator):
    super(NaAggregator, self).__init__()
    #aggregator, K = agg_str.split('_')
    if 'sage' == aggregator:
      self._op = SAGEConv(in_dim, out_dim, normalize=True)
    if 'gcn' == aggregator:
      self._op = GCNConv(in_dim, out_dim)
    if 'gat' == aggregator:
      heads = 8
      out_dim /= heads
      self._op = GATConv(in_dim, int(out_dim), heads=heads, dropout=0.5)
    if 'gin' == aggregator:
      nn1 = Sequential(Linear(in_dim, out_dim), ReLU(), Linear(out_dim, out_dim))
      self._op = GINConv(nn1)
    if aggregator in ['gat_sym', 'cos', 'linear', 'generalized_linear']:
      heads = 8
      out_dim /= heads
      self._op = GeoLayer(in_dim, int(out_dim), heads=heads, att_type=aggregator, dropout=0.5)
    if aggregator in ['sum', 'max']:
      self._op = GeoLayer(in_dim, out_dim, att_type='const', agg_type=aggregator, dropout=0.5)
    if aggregator in ['geniepath']:
      self._op = GeniePathLayer(in_dim, out_dim)

  def forward(self, x, edge_index):
    return self._op(x, edge_index)

class LaAggregator(nn.Module):

  def __init__(self, mode, hidden_size, num_layers=3):
    super(LaAggregator, self).__init__()
    self.jump = JumpingKnowledge(mode, channels=hidden_size, num_layers=num_layers)
    if mode == 'cat':
        self.lin = Linear(hidden_size * num_layers, hidden_size)
    else:
        self.lin = Linear(hidden_size, hidden_size)

  def forward(self, xs):
    return self.lin(F.relu(self.jump(xs)))

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class Zero(nn.Module):

  def __init__(self):
    super(Zero, self).__init__()

  def forward(self, x):
    return x.mul(0.)

