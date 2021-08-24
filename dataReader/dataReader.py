import torch
import dgl
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
from dgl.data.reddit import RedditDataset
from dgl.data.ppi import PPIDataset

def DGLDatasetReader(dataset_name,self_loops,device=None):

    data = load_data(dataset_name,self_loops)
    if dataset_name == 'reddit':
        g = data.graph.int().to(device)
        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1)


    else:
        g = DGLGraph(data.graph).to(device)
        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(device)
        g.ndata['norm'] = norm.unsqueeze(1)
    # add self loop
    if self_loops:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    return g,torch.FloatTensor(data.features),torch.LongTensor(data.labels),data.num_labels,\
           torch.ByteTensor(data.train_mask),torch.ByteTensor(data.test_mask),torch.ByteTensor(data.val_mask)

def load_data(dataset_name,self_loops):
    if dataset_name == 'cora':
        return citegrh.load_cora()
    elif dataset_name == 'citeseer':
        return citegrh.load_citeseer()
    elif dataset_name== 'pubmed':
        return citegrh.load_pubmed()
    elif dataset_name is not None and dataset_name.startswith('reddit'):
        return RedditDataset(self_loop=self_loops)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))
