import torch
from torch import nn
import numpy as np
from dgl import function as fn
from dgl.base import DGLError
from utils.utils_method import SimpleMultiLayerNN, eval_entropy



class SMG_mulithead(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=False,
                 n_head=3):
        super(SMG_mulithead,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self._cached = cached
        self._cached_h = None
        self.n_head=n_head
        self._k=k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self._alpha = nn.ParameterList()
        for i in range(self._k + 1):
            self._alpha.append(nn.Parameter(torch.Tensor(1)))
        self._lambada_fun = torch.nn.ModuleList()
        self._lambada_act = torch.nn.Softmax(dim=1)

        self._lambada_fun = torch.nn.ModuleList()
        for i in range(self.n_head):
            self._lambada_fun.append(nn.Linear(in_feats, 1, bias=True))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.out_feats)
        self.fc.weight.data.uniform_(-stdv, stdv)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        stdvk = 1. / np.sqrt(self._k)
        for i in range(self._k + 1):
            self._alpha[i].data.uniform_(-stdvk, stdvk)
        for hyper_model in self._lambada_fun:
            # xavier_normal_(hyper_model.weight)
            hyper_model.reset_parameters()

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat_list = self._cached_h
                result = torch.zeros(feat_list[0].shape[0], self.out_feats).to(feat_list[0].device)

                multi_g = {}
                for j in range(len(self._lambada_fun)):
                    multi_g[j] = []

                for i, k_feat in enumerate(feat_list):
                    for j, head_fun in enumerate(self._lambada_fun):
                        (multi_g[j]).append(head_fun(k_feat))

                g = torch.zeros(k_feat.shape[0], self._k + 1).to(self.device)
                entropy = torch.zeros(k_feat.shape[0]).to(self.device)
                for j, head_fun in enumerate(self._lambada_fun):
                    multi_g[j] = torch.cat(multi_g[j], axis=1)
                    entropy += eval_entropy(multi_g[j])
                    g += self._lambada_act(multi_g[j])


                for i, k_feat in enumerate(feat_list):
                    result += self.fc(k_feat * self._alpha[i] * g[:, i].unsqueeze(1))
            else:
                feat_list = []

                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X

                feat_list.append(feat.float())

                for i in range(self._k):
                    # norm = D^-1/2
                    feat = feat * norm
                    feat = feat.float()
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    # compute (D^-1 A^k D)^k X
                    feat = feat * norm
                    feat_list.append(feat)

                result = torch.zeros(feat_list[0].shape[0], self.out_feats).to(feat_list[0].device)

                multi_g = {}
                for j in range(len(self._lambada_fun)):
                    multi_g[j] = []

                for i, k_feat in enumerate(feat_list):
                    for j, head_fun in enumerate(self._lambada_fun):
                        (multi_g[j]).append(head_fun(k_feat))

                g = torch.zeros(k_feat.shape[0], self._k + 1).to(self.device)
                entropy = torch.zeros(k_feat.shape[0]).to(self.device)
                for j, head_fun in enumerate(self._lambada_fun):
                    multi_g[j] = torch.cat(multi_g[j], axis=1)
                    entropy += eval_entropy(multi_g[j])
                    g += self._lambada_act(multi_g[j])

                for i, k_feat in enumerate(feat_list):
                    result += self.fc(k_feat * self._alpha[i] * g[:, i].unsqueeze(1))

                if self.norm is not None:
                    result = self.norm(result)

                # cache feature
                if self._cached:
                    self._cached_h = feat_list
            return result, entropy





class SMG_mulithead_deep(SMG_mulithead):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=False,
                 n_head=3):
        super(SMG_mulithead_deep,self).__init__(in_feats,
                                      out_feats,
                                      k,
                                      cached,
                                      bias,
                                      norm,
                                      allow_zero_in_degree)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_head=n_head
        self.k=k
        self._lambada_fun = torch.nn.ModuleList()
        for i in range(self.n_head):
            self._lambada_fun.append(SimpleMultiLayerNN(in_feats, [k], 1,hidden_act_fun=torch.nn.Tanh()))
        self.reset_parameters()