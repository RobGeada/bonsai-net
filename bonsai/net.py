from bonsai.ops import *
from bonsai.helpers import *
from bonsai.cell import Cell


class Net(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # save creation params
        self.model_id = kwargs.get('model_id', namer())
        kwargs['model_id'] = self.model_id
        self.input_dim = kwargs['dim']
        self.data = None
        self.scale = kwargs['scale']
        self.patterns = looping_generator(kwargs['patterns'])
        self.nodes = kwargs['nodes']
        self.prune = {'edge':kwargs.get('prune', True), 'input':kwargs.get('prune',True)}
        self.classes = kwargs['classes']
        self.drop_prob = kwargs['drop_prob']
        self.towers = nn.ModuleDict()
        self.creation_params = kwargs

        # i/o
        self.log_print = print
        self.jn_print = print

        # training parameters
        self.lr_scheduler = LRScheduler(kwargs['lr_schedule']['T'], kwargs['lr_schedule']['lr_max'])

        # initialize torch params
        self.initializer = initializer(self.input_dim[1], 2 ** self.scale)
        dim = channel_mod(self.input_dim, 2 ** self.scale)
        self.cell_size_est = 0
        self.dim, self.dims, self.dim_counts = dim, [dim], []
        self.raw_cells, scalers = [], {}
        self.reductions = 0
        self.cells, self.cell_types, self.edges = None, {}, None
        if any(self.prune.values()):
            self.edge_pruners, self.input_pruners, self.edge_p_tot, self.input_p_tot = None, None, None, None
        self.global_pooling = None

        # build cells
        self.cell_idx = 0
        cell_dim_set = cell_dims(self.dim, self.scale, looping_generator(kwargs['patterns']))
        size_set = compute_sizes()
        if not all([dim in size_set for dim in cell_dim_set]):
            size_set = compute_sizes(cell_dim_set)
        self.size_set = size_set

        for pattern in range(kwargs['num_patterns']):
            self.add_pattern()

    def add_cell(self, cell_type, aux, prune=None, scale=False, full_ops=False):
        if prune is None:
            prune = self.prune
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
                                   genotype=cell_genotype,
                                   op_sizes=self.size_set,
                                   random_ops=random_ops,
                                   prune=prune))

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

    def add_pattern(self, prune=None, full_ops=False):
        if prune is None:
            prune = self.prune
        prune = {'edge':prune,'input':prune}
        for cell in next(self.patterns):
            cell_type = 'Normal' if 'n' in cell else 'Reduction'
            aux = 'a' in cell
            scale = 's' in cell
            self.add_cell(cell_type, prune=prune, aux=aux, scale=scale, full_ops=full_ops)

    def track_params(self):
        self.cells = nn.ModuleList(self.raw_cells)
        self.cell_size_est = sum([cell.size_est for cell in self.cells])
        self.edges = nn.ModuleList([cell.edges[key] for cell in self.cells for key in cell.edges])
        if any(self.prune.values()):
            if self.prune['edge']:
                self.edge_pruners = [pruner for cell in self.cells for pruner in cell.edge_pruners]
                self.edge_p_tot = torch.Tensor([len(commons) * len(cell.edges) for cell in self.cells]).cuda()
                self.edge_sizes = []
                for cell in self.cells:
                    cell_sizes = sum([op.pruner.mem_size for k, edge in cell.edges.items() for op in edge.ops])
                    self.edge_sizes.append(cell_sizes)
                self.edge_sizes = torch.Tensor(self.edge_sizes).cuda()

            if self.prune['input']:
                self.input_pruners= [pruner for cell in self.cells for pruner in cell.input_pruners]
                self.input_p_tot = torch.Tensor(self.dim_counts).cuda()
            self.save_genotype()
        self.global_pooling = nn.AdaptiveAvgPool2d(self.dim[1:][-1])

    def remove_pruners(self, remove_input=False, remove_edge=False):
        self.prune = {'edge': self.prune['edge'] and not remove_edge, 'input': self.prune['input'] and not remove_input}
        for cell in self.cells:
            if remove_input:
                cell.input_handler.prune = False
                del cell.input_handler.pruners
                del cell.input_pruners
                cell.prune['input']=False
            if remove_edge:
                del cell.edge_pruners
                cell.prune['edge'] = False
                for k,edge in cell.edges.items():
                    del edge.pruners
                    for op in edge.ops:
                        op.prune = False
                        del op.pruner
       
        if remove_edge:
            del self.edge_pruners
            self.edge_pruners=[]

        if remove_input:
            del self.input_pruners
            del self.input_p_tot

        if remove_input:
            clean("Input Pruner Removal")
        if remove_edge:
            clean("Edge Pruner Removal")

    def forward(self, x, drop_prob=None, auxiliary=False, verbose=False):
        if drop_prob is None:
            drop_prob = self.drop_prob
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
        if self.edge_pruners:
            deadheads += sum([edge.deadhead() for edge in self.edges])
        if self.input_pruners:
            deadheads += sum([cell.input_handler.deadhead() for cell in self.cells])
        self.log_print("\nDeadheaded {} operations".format(deadheads))
        self.log_print("Param Delta: {:,} -> {:,}".format(old_params, general_num_params(self)))
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
                if self.prune['edge']:
                    edge_params = [[str(op), conversion(op.pruner.weight)] for op in edge.ops if not op.zero]
                else:
                    edge_params = [[str(op), 1] for op in edge.ops if not op.zero]
                cell_genotype["{}->{}".format(edge.origin, edge.target)] = edge_params

            # input genotype
            if self.prune['input']:
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
        soft_ops = 0
        hard_ops = 0
        out = [None,None,None]
        if self.prune['edge']:
            for cell in self.cells:
                soft_ops, hard_ops = cell.genotype_compression(soft_ops, hard_ops)
            out[0] = (hard_ops/sum(self.edge_sizes)).item()
            out[1] = (soft_ops/sum(self.edge_sizes)).item()
        else:
            useds, posses = 0, 0
            for cell in self.cells:
                used, poss = cell.genotype_compression(soft_ops, hard_ops, used_ratio=True)
                useds += used
                posses += poss
            out[0] = useds/posses
        if self.prune['input']:
            out[2] = np.mean([len(cell.input_handler.get_ins()) for cell in self.cells])
        return out

    def __str__(self):
        def out_format(l="", p="", d="", c=""):
            sep = ' : '
            out_fmt = '{l:}{s}{d}{s}{p}{s}{c}\n'.format(l='{l:<20}', d='{d:^12}', p='{p:^10}', c='{c:^9}', s=sep)
            p = "{:,}".format(p) if isinstance(p, (float, int)) else p
            c = "{:.1f}%".format(c*100) if isinstance(c, (float, int)) else c
            c = "" if c is None else c
            return out_fmt.format(l=l, p=p, d=d, c=c)

        spacer = '{{:=^{w}}}\n'.format(w=len(out_format()))
        out = spacer.format(" NETWORK ")
        out += spacer.format(" "+self.model_id+" ")
        out += out_format(l='', d='Dim', p='Params', c='Comp')

        out += out_format(l="Initializer", p=general_num_params(self.initializer))
        for i,cell in enumerate(self.cells):
            out += cell.__repr__(out_format)
            if str(i) in self.towers.keys():
                out += out_format(l=" ↳ Aux Tower", p=general_num_params(self.towers[str(i)]))
        if 'Classifier' in self.towers:
            out += out_format(l=" ↳ Classifier", p=general_num_params(self.towers['Classifier']))
        out += spacer.format("")
        out += out_format(l="Total", p=general_num_params(self), c=self.genotype_compression()[0])
        out += spacer.format("")
        return out

    def creation_string(self):
        return "ID: '{}', Dim: {}, Classes: {}, Scale: {}, N: {}, Patterns: {}, Nodes: {}, Pruners: {}".format(
            self.model_id,
            self.input_dim,
            self.classes,
            self.scale,
            len(self.dims)-1,
            self.patterns,
            self.nodes,
            self.prune)
    
    def detail_print(self, minimal=False):
        self.jn_print('==================== NETWORK ===================')
        self.jn_print(self.creation_string())
        self.jn_print("Cell Size Est:",sizeof_fmt(self.cell_size_est*1024*1024))
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
