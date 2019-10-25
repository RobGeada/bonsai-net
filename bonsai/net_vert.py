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
            if random_ops.get('worst_case',False):
                test_ops = {k:v(dim[1],stride) for k,v in commons.items()}
                op_params = {k:general_num_params(v) for k,v in test_ops.items()}
                del test_ops
                used,used_ops,total = 0, {}, sum(op_params.values())
                available_ops = {}
                for k,v in op_params.items():
                    if np.random.rand() < (used/total):
                        available_ops[k]=commons[k]
                        used_ops[k]=v
                        used+=v
                print(used_ops,sum(used_ops.keys()))



                if len(available_ops) and largest not in available_ops.keys():
                    available_ops.pop(list(available_ops.keys())[0])
                    available_ops[largest]=commons[largest]
            else:
                available_ops = {k:v for k,v in commons.items() if np.random.rand() < random_ops['e_c']}
            op_weights = [1.]*len(available_ops)
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

        self.mask_gen = lambda dim, drop_prob: Variable(torch.cuda.FloatTensor(dim, 1, 1, 1).bernoulli(1-drop_prob))

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

    def set_half_mask(self):
        self.mask_gen = lambda dim, drop_prob: Variable(torch.cuda.HalfTensor(dim, 1, 1, 1).bernoulli(1 - drop_prob))

    def forward(self, x, drop_prob):
        if self.num_ops:
            summed = self.normalizer(sum([op(x) for op in self.ops]))
        else:
            # zero out input directly
            summed = self.normalizer(x)

        if random.random() < drop_prob:
            summed = summed.div(1-drop_prob)
            summed = summed.mul(self.mask_gen(summed.size(0), drop_prob))
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
        if self.cell_type == 'Reduction':
            dim_rep = self.in_dim[1],self.in_dim[2]//2
        else:
            dim_rep = self.in_dim[1:3]
        out = "Cell {:<2} {:<12}: {}, {:>11,} params".format(
            self.name,
            '(' + self.cell_type + ")",
            '{:^4}x{:^4}'.format(*dim_rep),
            general_num_params(self))
        return out

    def detail_print(self, minimal):
        out = ""
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


class Net(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # save creation params
        self.model_id = kwargs.get('model_id', namer())
        kwargs['model_id'] = self.model_id
        self.input_dim = kwargs['dim']
        self.scale = kwargs['scale']
        self.patterns = looping_generator(kwargs['patterns'])
        self.nodes = kwargs['nodes']
        self.prune = kwargs.get('prune', True)
        self.auxiliary = kwargs.get('auxiliary', False)
        self.classes = kwargs['classes']
        self.towers = nn.ModuleDict()
        self.at_max_depth = False
        self.creation_params = kwargs
        self.half = False

        # i/o
        self.log_print = print
        self.jn_print = print

        # initialize params
        self.initializer = initializer(self.input_dim[1], 2 ** self.scale)
        dim = channel_mod(self.input_dim, 2 ** self.scale)
        self.dim, self.dims, self.dim_counts = dim, [dim], []
        self.raw_cells, scalers = [], {}
        self.reductions = 0
        self.cells, self.cell_types, self.edges = None, {}, None
        if self.prune:
            self.edge_pruners, self.input_pruners, self.edge_p_tot, self.input_p_tot = None, None, None, None
        self.global_pooling = None

        # build cells
        self.cell_idx = 0
        for pattern in range(kwargs['num_patterns']):
            self.add_pattern(init=True)

    def add_cell(self, cell_type, aux, init=False, scale=False, full_ops=False):
        cell_type = 'Normal' if len(self.dims) == 1 else cell_type
        random_ops = False if full_ops else self.creation_params.get("random_ops", False)
        if 'genotype' in self.creation_params:
            cell_genotype = self.creation_params['genotype'].get(len(self.dims)-1)
        else:
            cell_genotype = None

        self.raw_cells.append(Cell(len(self.dims)-1,
                                   cell_type,
                                   self.dims,
                                   self.nodes,
                                   cell_genotype,
                                   random_ops=random_ops,
                                   prune=self.prune))

        if cell_type is 'Reduction':
            self.dim = cw_mod(self.dim, 2)
            self.reductions += 1
        self.dim_counts += [len(self.dims)]
        self.dims.append(self.dim)

        if aux:
            if len(self.towers) and 'Classifier' in self.towers:
                new_name = str(self.towers['Classifier'].position)
                self.towers[new_name] = self.towers.pop('Classifier')
            self.towers['Classifier'] = Classifier(len(self.dims)-2, 
                                                   self.dim, 
                                                   self.classes, 
                                                   scale=scale)
        self.track_params()
        return True

    def add_pattern(self, init=False, full_ops=False):
        for cell in next(self.patterns):
            cell_type = 'Normal' if 'n' in cell else 'Reduction'
            aux = 'a' in cell
            scale = 's' in cell
            self.add_cell(cell_type, aux=aux, scale=scale, init=init, full_ops=full_ops)

    def track_params(self):
        self.cells = nn.ModuleList(self.raw_cells)
        self.edges = nn.ModuleList([cell.edges[key] for cell in self.cells for key in cell.edges])
        if self.prune:
            self.edge_pruners = [pruner for cell in self.cells for pruner in cell.edge_pruners]
            self.input_pruners= [pruner for cell in self.cells for pruner in cell.input_pruners]
            self.edge_p_tot = torch.Tensor([len(commons)*len(cell.edges) for cell in self.cells]).cuda()
            self.edge_params = []
            for cell in self.cells:
                cell_params = sum([op.pruner.params for k,edge in cell.edges.items() for op in edge.ops])
                self.edge_params.append(cell_params)
            self.edge_params = torch.Tensor(self.edge_params).cuda()
            self.input_p_tot = torch.Tensor(self.dim_counts).cuda()
            self.save_genotype()
        self.global_pooling = nn.AdaptiveAvgPool2d(self.dim[1:][-1])

    def remove_pruners(self):
        self.prune = False
        for cell in self.cells:
            cell.input_handler.prune = False
            del cell.input_handler.pruners
            del cell.edge_pruners
            del cell.input_pruners

            for k,edge in cell.edges.items():
                del edge.pruners
                for op in edge.ops:
                    op.prune = False
                    del op.pruner
        del self.edge_pruners
        del self.input_pruners
        del self.edge_p_tot
        del self.input_p_tot
        clean("Pruner Removal")

    def forward(self, x, drop_prob=0, auxiliary=False, verbose=False):
        outputs = []
        xs = [self.initializer(x)]
        if verbose:
            self.jn_print("Init: {}".format(mem_stats()))
        for i in range(len(self.cells)):
            x = self.cells[i].forward(xs, drop_prob)
            if verbose:
                self.jn_print("{}: {}".format(i, mem_stats()))
            if str(i) in self.towers.keys():
                outputs.append(self.towers[str(i)](x))
                if verbose:
                    self.jn_print("Tower {}: {}".format(i, mem_stats()))
            xs.append(x)
        x = self.global_pooling(xs[-1])
        if verbose:
            self.jn_print("{}: {}".format('GP', mem_stats()))
        outputs.append(self.towers['Classifier'](x))
        if verbose:
            self.jn_print("Classifier: {}".format(mem_stats()))
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
        ops = 0
        for cell in self.cells:
            for key, edge in cell.edges.items():
                for op in edge.ops:
                    ops += general_num_params(op.op)
        edge_comp = ops/sum(self.edge_params)
        input_comp = np.mean([len(cell.input_handler.get_ins()) for cell in self.cells])
        return edge_comp.item()   , input_comp

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
                out += " ↳ Aux Tower        :            {:>11,} params\n".format(
                        general_num_params(self.towers[str(i)]))
        if 'Classifier' in self.towers:
            out += \
                " ↳ Classifier       :            {:>11,} params".format(general_num_params(self.towers['Classifier']))
        out += "\n{}\n".format("".join(["="]*total_spacer))
        out += "Total               :            {:>11,} params".format(general_num_params(self))
        out += "\n{}\n".format("".join(["="]*total_spacer))
        return out

    def creation_string(self):
        return "ID: '{}', Dim: {}, Classes: {}, Scale: {}, N: {}, Patterns: {}, Nodes: {}, Pruners: {}, Aux: {}".format(
            self.model_id,
            self.input_dim,
            self.classes,
            self.scale,
            len(self.dims)-1,
            self.patterns,
            self.nodes,
            self.prune,
            self.auxiliary)
    
    def detail_print(self, minimal=False):
        self.jn_print('==================== NETWORK ===================')
        self.jn_print(self.creation_string())
        self.jn_print('Initializer              : {:>10,} params'.format(general_num_params(self.initializer)))
        for i, cell in enumerate(self.cells):
            self.jn_print("=== {} ===".format(cell))
            self.jn_print(cell.detail_print(minimal))
            if str(i) in self.towers.keys():
                self.jn_print('Aux Tower:               : {:>10,} params'.format(
                    general_num_params(self.towers[str(i)])))
        self.jn_print("Classifier               : {:>10,} params".format(general_num_params(self.towers['Classifier'])))
        self.jn_print("================================================")
        self.jn_print("Total                    : {:>10,} params".format(general_num_params(self)))
        self.jn_print("================================================")
