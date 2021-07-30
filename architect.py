import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect(object):

    def __init__(self, model, args):
        self.args = args
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, data, eta, network_optimizer):
        loss = self.model._loss(data, is_valid=False) #train loss
        theta = _concat(self.model.parameters()).data# w
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta#gradient, L2 norm
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta)) # one-step update, get w' for Eq.7 in the paper
        return unrolled_model

    def step(self, data, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(data, eta, network_optimizer)
        else:
            self._backward_step(data, is_valid=True)
        self.optimizer.step()

    def _backward_step(self, data, is_valid=True):
        if self.args.data == 'PPI':
            device = torch.device('cuda:%d' % self.args.gpu if torch.cuda.is_available() else 'cpu')
            for valid_data in data[1]:
                valid_data = valid_data.to(device)
                loss = self.model._loss_ppi(valid_data, is_valid)
                loss.backward()
        else:
            loss = self.model._loss(data, is_valid)
            loss.backward()

    def _backward_step_unrolled(self, data, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(data, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(data, is_valid=True) # validation loss

        unrolled_loss.backward() # one-step update for w?
        dalpha = [v.grad for v in unrolled_model.arch_parameters()] #L_vali w.r.t alpha
        vector = [v.grad.data for v in unrolled_model.parameters()] # gradient, L_train w.r.t w, double check the model construction
        implicit_grads = self._hessian_vector_product(vector, data)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        #update alpha, which is the ultimate goal of this func, also the goal of the second-order darts
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, data, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v) # R * d(L_val/w', i.e., get w^+
        loss = self.model._loss(data, is_valid=False) # train loss
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters()) # d(L_train)/d_alpha, w^+

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2*R, v) # get w^-, need to subtract 2 * R since it has add R
        loss = self.model._loss(data, is_valid=False)# train loss
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())# d(L_train)/d_alpha, w^-

        #reset to the orignial w, always using the self.model, i.e., the original model
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
