import random
import pickle as pkl
from torch.autograd import Variable

from bonsai.ops import *
from bonsai.helpers import *


class Edge(nn.Module):
    def __init__(self, dim, origin, target, stride=1, genotype=None, random_ops=False, prune=True):
        super().__init__()
        self.stride = stride
        if genotype is not None:
            available_ops = {k: commons[k] for (k, weight) in genotype}
            op_weights = [weight for (k, weight) in genotype]
        elif random_ops is not False:
            available_ops = {k:v for k,v in commons.items() if np.random.rand() < random_ops[0]}
            op_weights = [1]*len(available_ops)
        else:
            available_ops = commons
            op_weights = [None]*len(commons)

        self.ops = nn.ModuleList(
                [PrunableOperation(op, key, dim[1], stride=stride, pruner_init=op_weights[i], prune=prune)
                 for i, (key, op) in enumerate(available_ops.items())])

        if prune:
            self.pruners = [op.pruner for op in self.ops]

        self.num_ops = len([op for op in self.ops if not op.zero])
        self.dim = dim
        self.origin = origin
        self.target = target

        if self.num_ops:
            self.normalizer = nn.BatchNorm2d(dim[1])
        else:
            self.normalizer = Zero(stride=self.stride)

    def deadhead(self):
        dhs = sum([op.deadhead() for i, op in enumerate(self.ops)])
        self.num_ops -= dhs
        if self.num_ops == 0:
            self.normalizer = Zero(stride=self.stride)
        return dhs

    def __str__(self):
        return "{}->{}: {}, \t({:,} params)".format(
            self.origin,
            self.target,
            [str(op) for op in self.ops if (not op.zero)],
            general_num_params(self))

    def __repr(self):
        return str(self)

    def forward(self, x, drop_prob):
        if self.num_ops:
            summed = self.normalizer(sum([op(x) for op in self.ops]))
        else:
            # zero out input directly
            summed = self.normalizer(x)

        if random.random() < drop_prob:
            mask = Variable(torch.cuda.FloatTensor(summed.size(0), 1, 1, 1).bernoulli(1-drop_prob))
            summed = summed.div(1-drop_prob)
            summed = summed.mul(mask)
        return summed


class Cell(nn.Module):
    def __init__(self, name, cell_type, dims, nodes, genotype=None, random_ops=False, prune=True):
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
        for origin in ['x', 'y']:
            for target in range(nodes):
                key = "{}->{}".format(origin, target)
                edges.append([key, Edge(self.in_dim,
                                        origin,
                                        target,
                                        stride=1 if cell_type == 'Normal' else 2,
                                        random_ops=random_ops,
                                        genotype=None if genotype is None else genotype.get(key),
                                        prune=prune)])
                keys[key] = {'origin': origin, 'target': target}

        # connect data nodes
        for origin in range(nodes):
            for target in range(origin+1, nodes):
                key = "{}->{}".format(origin, target)
                edges.append([key, Edge(self.in_dim,
                                        origin,
                                        target,
                                        random_ops=random_ops,
                                        genotype=None if genotype is None else genotype.get(key),
                                        prune=prune)])
                keys[key] = {'origin': origin, 'target': target}

        self.node_names = ['x', 'y']+list(range(self.nodes))
        self.normalizers = nn.ModuleDict({str(k): normalizer(self.in_dim[1]) for k in self.node_names})

        if prune:
            self.edge_pruners = [pruner for key, edge in edges for pruner in edge.pruners]
            self.input_pruners= [pruner for pruner in self.input_handler.pruners]
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

    def __repr__(self):
        out = "Cell {:<2} {:<12}: {}, {:>11,} params".format(
            self.name,
            '(' + self.cell_type + ")",
            '{:^4}x{:^4}'.format(*self.in_dim[1:3]),
            general_num_params(self))
        return out

    def detail_print(self):
        out = ""
        out += "X: {}, ({:,} params) \n".format(self.name-1 if self.name else 'In',
                                                general_num_params(self.scaler))
        out += "Y: {}, ({:,} params)\n".format(self.input_handler,
                                               general_num_params(self.input_handler))
        for key, edge in self.edges.items():
            out += '    {}\n'.format(edge)
        return out

    def get_parameters(self, selector):
        params = []
        for key, edge in self.edges.items():
            params += edge.get_parameters(selector)
        return params


class Net(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # save creation params
        self.model_id = kwargs.get('model_id', namer())
        kwargs['model_id'] = self.model_id
        self.input_dim = kwargs['dim']
        self.scale = kwargs['scale']
        self.spacing = kwargs['spacing']
        self.reductions = kwargs['reductions']
        self.nodes = kwargs['nodes']
        self.prune = kwargs.get('prune',True)
        self.auxiliary = kwargs.get('auxiliary',False)
        self.classes = kwargs['classes']
        self.towers = nn.ModuleDict()
        self.creation_params = kwargs

        # i/o
        self.log_print = print
        self.jn_print = print

        # initialize params
        self.initializer = initializer(self.input_dim[1], 2 ** self.scale)
        dim = channel_mod(self.input_dim, 2 ** self.scale)
        cells, scalers = [], {}

        # build cells
        dims = [dim]
        dim_counts = []
        for r in range(self.reductions):
            for s in range(self.spacing):
                cell_name = r*self.spacing+s
                cell_genotype = None if kwargs.get('genotype') is None else kwargs.get('genotype').get(cell_name)
                if s != self.spacing - 1:
                    cells.append(Cell(cell_name,
                                      'Normal',
                                      dims,
                                      self.nodes,
                                      cell_genotype,
                                      random_ops=kwargs.get("random_ops", False),
                                      prune=self.prune))
                    dim_counts += [len(dims)]
                    dims.append(dim)
                else:
                    cells.append(Cell(cell_name,
                                      'Reduction',
                                      dims,
                                      self.nodes,
                                      cell_genotype,
                                      random_ops=kwargs.get("random_ops", False),
                                      prune=self.prune))
                    dim = cw_mod(dim, 2)
                    if self.auxiliary and r<=self.reductions-1:
                        self.towers[str(cell_name)] = Classifier(dim, self.classes)
                    dim_counts += [len(dims)]
                    dims.append(dim)
                    
        # put all parameters into Torch
        self.cells = nn.ModuleList(cells)
        self.edges = nn.ModuleList([cell.edges[key] for cell in self.cells for key in cell.edges])
        if self.prune:
            self.edge_pruners = [pruner for cell in self.cells for pruner in cell.edge_pruners]
            self.input_pruners= [pruner for cell in self.cells for pruner in cell.input_pruners]
            self.edge_p_tot = torch.Tensor([len(commons)*len(cell.edges) for cell in self.cells]).cuda()
            self.input_p_tot = torch.Tensor(dim_counts).cuda()
            self.save_genotype()
        self.global_pooling = nn.AdaptiveAvgPool2d(dim[1:][-1])
        self.towers['Classifier'] = Classifier(dim, self.classes)

    def forward(self, x, drop_prob=0, auxiliary=False, verbose=False):
        outputs = []
        xs = [self.initializer(x)]
        for i in range(len(self.cells)):
            if verbose:
                self.jn_print("{}: {}".format(i, mem_stats()))
            x = self.cells[i].forward(xs, drop_prob)
            if str(i) in self.towers.keys():
                outputs.append(self.towers[str(i)](x))
            xs.append(x)
        x = self.global_pooling(xs[-1])
        outputs.append(self.towers['Classifier'](x))
        return outputs if auxiliary else outputs[-1]

    def deadhead(self):
        old_params = general_num_params(self)
        deadheads = 0
        deadheads += sum([edge.deadhead() for edge in self.edges])
        deadheads += sum([cell.input_handler.deadhead() for cell in self.cells])
        self.log_print("\nDeadheaded {} operations".format(deadheads))
        self.log_print("Param Delta: {:,} -> {:,}".format(old_params, general_num_params(self)))
        #self.detail_print()
        self.save_genotype()

    def extract_genotype(self, tensors=True):
        if tensors:
            def conversion(x): return x
        else:
            def conversion(x): return np.round(x.item(), 2)

        cell_genotypes = {}
        for cell in self.cells:
            cell_genotype = {}

            # edge genotype
            for key, edge in cell.edges.items():
                if self.prune:
                    edge_params = [[str(op), conversion(op.pruner.weight)] for op in edge.ops if not op.zero]
                else:
                    edge_params = [[str(op), 1] for op in edge.ops if not op.zero]
                cell_genotype["{}->{}".format(edge.origin, edge.target)] = edge_params

            # input genotype
            if self.prune:
                cell_genotype['Y'] = {'weights': [conversion(pruner.weight) for pruner in cell.input_handler.pruners],
                                      'zeros': cell.input_handler.zeros}
            else:
                cell_genotype['Y'] = {'zeros': cell.input_handler.zeros}
            cell_genotypes[cell.name] = cell_genotype

        return self.creation_params, cell_genotypes

    def save_genotype(self):
        id_str = self.model_id.replace(" ","_")
        pkl.dump(self.extract_genotype(), open('genotypes/genotype_{}.pkl'.format(id_str), 'wb'))
        pkl.dump(self.extract_genotype(tensors=False), open('genotypes/genotype_{}_np.pkl'.format(id_str), 'wb'))

    def genotype_compression(self):
        ops, potential = 0, 0
        for cell in self.cells:
            for key, edge in cell.edges.items():
                ops += edge.num_ops
                potential += len(commons)
        edge_comp = ops/potential
        input_comp = np.mean([len(cell.input_handler.get_ins()) for cell in self.cells])
        return edge_comp, input_comp

    def scale_up(self):
        self.jn_print("\x1b[31mScaling from {} to {}\x1b[0m".format(self.scale, self.scale+1))
        self.cpu()
        creation_params, genotype = self.extract_genotype()
        creation_params['scale'] += 1
        creation_params['dim'] = batch_mod(creation_params['dim'], 2)
        creation_params['genotype'] = genotype
        self.__init__(**creation_params)
        self.cuda()

    def __str__(self):
        spacers = 50
        net_spacers = "".join(["="] * ((spacers - len("NETWORK")) // 2))
        name_spacers = "".join([" "]*((spacers-len(self.model_id))//2))
        total_spacer = 2*len(name_spacers)+len(self.model_id)+2

        out = '{} NETWORK {}\n'.format(net_spacers, net_spacers)
        out += '{} {} {}\n'.format(name_spacers,self.model_id,name_spacers)
        out += 'Initializer         :            {:>11,} params\n'.format(general_num_params(self.initializer))
        for i,cell in enumerate(self.cells):
            out += str(cell) + "\n"
            if str(i) in self.towers.keys():
                out += " ↳ Aux Tower       :            {:>11,} params\n".format(
                        general_num_params(self.towers[str(i)]))
        out += "Classifier          :            {:>11,} params".format(general_num_params(self.towers['Classifier']))
        out += "\n{}\n".format("".join(["="]*total_spacer))
        out += "Total               :            {:>11,} params".format(general_num_params(self))
        out += "\n{}\n".format("".join(["="]*total_spacer))
        return out

    def creation_string(self):
        return "ID: '{}', Dim: {}, Classes: {}, Scale: {}, Spacing: {}, Reductions: {}, Nodes: {}, Pruners: {}, Aux: {}".format(
            self.model_id,
            self.input_dim,
            self.classes,
            self.scale,
            self.spacing,
            self.reductions,
            self.nodes,
            self.prune,
            self.auxiliary)
    
    def detail_print(self):
        self.jn_print('==================== NETWORK ===================')
        self.jn_print(self.creation_string())
        self.jn_print('Initializer              : {:>10,} params'.format(general_num_params(self.initializer)))
        for i, cell in enumerate(self.cells):
            self.jn_print("=== {} ===".format(cell))
            self.jn_print(cell.detail_print())
            if str(i) in self.towers.keys():
                self.jn_print('Aux Tower:               : {:>10,} params'.format(
                    general_num_params(self.towers[str(i)])))
        self.jn_print("Classifier               : {:>10,} params".format(general_num_params(self.towers['Classifier'])))
        self.jn_print("================================================")
        self.jn_print("Total                    : {:>10,} params".format(general_num_params(self)))
        self.jn_print("================================================")
