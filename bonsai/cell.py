from bonsai.ops import *
from bonsai.helpers import *
from bonsai.edge import Edge


class Cell(nn.Module):
    def __init__(self, name, cell_type, dims, nodes, op_sizes, genotype=None, random_ops=False, prune=True):
        super().__init__()
        self.name = name
        self.cell_type = cell_type
        self.nodes = nodes
        self.dims = dims
        self.input_handler = PrunableInputs(dims,
                                            scale_mod=1 if self.cell_type is 'Normal' else 2,
                                            genotype={} if genotype is None else genotype['Y'],
                                            random_ops=random_ops,
                                            prune=prune)
        self.in_dim = channel_mod(dims[-1], dims[-1][1] if cell_type == 'Normal' else dims[-1][1]*2)
        self.scaler = MinimumIdentity(dims[-1][1],
                                      dims[-1][1] * (2 if self.cell_type is 'Reduction' else 1),
                                      stride=1)

        edges = []
        keys = {}

        # link input data to each antecedent node
        self.cell_size_est = 0
        if random_ops:
            cell_cnx = sum([1 for origin in ['x', 'y'] for target in range(nodes)])
            cell_cnx += sum([1 for origin in range(nodes) for target in range(origin + 1, nodes)])
            aim_size = cell_cnx*sum(op_sizes[self.in_dim].values())*random_ops['e_c']
        else:
            aim_size = None
        for origin in ['x', 'y']:
            for target in range(nodes):
                key = "{}->{}".format(origin, target)
                edges.append([key, Edge(self.in_dim,
                                        origin,
                                        target,
                                        stride=1 if cell_type == 'Normal' else 2,
                                        aim_size=aim_size,
                                        op_sizes=op_sizes[self.in_dim],
                                        genotype=None if genotype is None else genotype.get(key),
                                        prune=prune)])
                if aim_size:
                    aim_size -= edges[-1][1].used
                keys[key] = {'origin': origin, 'target': target}

        # connect data nodes
        for origin in range(nodes):
            for target in range(origin+1, nodes):
                key = "{}->{}".format(origin, target)
                edges.append([key, Edge(self.in_dim,
                                        origin,
                                        target,
                                        aim_size=aim_size,
                                        op_sizes=op_sizes[self.in_dim],
                                        genotype=None if genotype is None else genotype.get(key),
                                        prune=prune)])
                if aim_size:
                    aim_size -= edges[-1][1].used
                keys[key] = {'origin': origin, 'target': target}
        self.size_est = sum([edge[1].used for edge in edges])
        self.node_names = ['x', 'y']+list(range(self.nodes))
        self.normalizers = nn.ModuleDict({str(k): normalizer(self.in_dim[1]) for k in self.node_names})

        self.prune = prune
        if prune['edge']:
            self.edge_pruners = [pruner for key, edge in edges for pruner in edge.pruners]
        if prune['input']:
            self.input_pruners  = [pruner for pruner in self.input_handler.pruners]
        self.edges = nn.ModuleDict(edges)
        self.key_ots = dict([(k,(v['origin'], v['target'])) for (k, v) in keys.items()])
        self.keys_by_origin = dict([(i, [k for (k, v) in keys.items() if i == v['origin']]) for i in self.node_names])
        self.keys_by_target = dict([(i, [k for (k, v) in keys.items() if i == v['target']]) for i in self.node_names])
        self.genotype_width = len(commons.items())

    def forward(self, xs, drop_prob):
        outs = {}
        for node in self.node_names:
            if node == 'x':
                raw_node_in = [self.scaler(xs[-1])]
            elif node == 'y':
                raw_node_in = [self.input_handler(xs)]
            else:
                raw_node_in = [outs[origin] for origin in self.keys_by_target[node]]
            node_in = self.normalizers[str(node)](sum(raw_node_in))
            if node == self.nodes-1:
                del outs
                return node_in
            else:
                for key in self.keys_by_origin[node]:
                    outs[key] = self.edges[key](node_in, drop_prob)

    def genotype_compression(self, soft_ops=0, hard_ops=0, used_ratio=False):
        if not used_ratio:
            for key, edge in self.edges.items():
                soft_ops, hard_ops = edge.genotype_compression(soft_ops, hard_ops)
            return soft_ops, hard_ops
        else:
            used, possible = 0, 0
            for key, edge in self.edges.items():
                possible += edge.possible
                used += edge.used
            return used, possible

    def __repr__(self, out_format):
        if self.cell_type == 'Reduction':
            dim_rep = self.in_dim[1],self.in_dim[2]//2
        else:
            dim_rep = self.in_dim[1:3]
        comp = self.genotype_compression(used_ratio=True)
        comp = comp[0]/comp[1]
        layer_name = "Cell {:<2} {:<12}".format(self.name,'(' + self.cell_type + ")")
        dim = '{:^4}x{:^4}'.format(*dim_rep)
        out = out_format(l=layer_name, d=dim, p=general_num_params(self), c=comp)
        return out

    def detail_print(self, minimal):
        out = ""
        out += 'Size Est: {}\n'.format(sizeof_fmt(self.size_est*1024*1024))
        out += "X: {}, ({:,} params) \n".format(self.name-1 if self.name else 'In',
                                                general_num_params(self.scaler))
        out += "Y: {}, ({:,} params)\n".format(self.input_handler,
                                               general_num_params(self.input_handler))
        if not minimal:
            for key, edge in self.edges.items():
                out += '    {}\n'.format(edge)
        return out

    def get_parameters(self, selector):
        params = []
        for key, edge in self.edges.items():
            params += edge.get_parameters(selector)
        return params