from bonsai.data_loaders import load_data
from bonsai.net import Net
from bonsai.trainers import *
from bonsai.helpers import *
from bonsai.ops import *


# === I/O ==============================================================================================================
def jn_print(x,end="\n"):
    print(x,end=end)
    with open("logs/jn_out.log", "a") as f:
        f.write(x+end)


# === INITIALIZATIONS ==================================================================================================
def gen_compression_targets(hypers):
    sizes = {}
    print("Search Range: {:.2f}->{:.2f}".format(1 / len(commons), 1.))
    init_size = sp_size_test(1, e_c=1, add_pattern=0, remove_prune=False, **hypers)
    jn_print('First pattern: {}'.format(init_size[0]))
    if init_size[0]>hypers['gpu_space']:
        jn_print("First pattern too large for GPU!")
        return [], 0
    
    for n in range(1, hypers['total_patterns']):
        sizes[n] = []
        bst = BST(1 / (len(commons)), 1., depth=6)
        while bst.answer is None:
            print("{}: {:.3f}\r".format(n, bst.pos), end="")
            queries = []
            for q in range(1):
                size = sp_size_test(n, e_c=bst.pos, add_pattern=1, remove_prune=False, **hypers)
                queries.append(not (not size[1] and (size[0]) < hypers['gpu_space']))
            bst.query(any(queries))
        if bst.passes:
            sizes[n] = max(bst.passes)
        print()

    if any([v for (k, v) in sizes.items() if v == 1]):
        start_size = [k for (k, v) in sizes.items() if v == 1][-1] + 1
    else:
        start_size = 1

    jn_print("Comp Ratios:\nsizes={{\n{}\n}},".format("\n".join(["    {}:{},".format(k, v) for (k, v) in sizes.items()])))
    jn_print("start_size={}".format(start_size))
    jn_print("Effective Scale: {:.2f}".format(hypers['scale']*sizes[max(sizes.keys())]*len(commons)))
    return sizes, start_size


# === MODEL CLASS ======================================================================================================
class Bonsai:
    def __init__(self, hypers, sizes=None, start_size=None):
        wipe_output()
        self.hypers = hypers
        self.model_id = namer()
        
        
        self.data, self.dim = load_data(hypers['batch_size'], hypers['dataset']['name'])
        
        if 'genotype' not in hypers:
            self.hypers['total_patterns'] = get_n_patterns(hypers['patterns'],
                                                           self.dim,
                                                           hypers['reduction_target']) + hypers['post_patterns']
            if sizes is None:
                jn_print("== Determining compression ratios ==")
                self.sizes, self.start_size = gen_compression_targets(hypers)
            else:
                self.sizes, self.start_size = sizes, start_size
        else:
            self.start_size = start_size
        self.model = None
        self.random = 0
        self.e_c, self.i_c = [], []
        
    # random levels:
    # 0: not random
    # 1: random at same level as penult. cell, after fully connected, prunable last cell
    # 2: random at same level as epoch 599, pruning
    # 3: random at same level as epoch 599, no pruning
    def generate_model(self):
        if self.random and self.e_c is None:
            raise ValueError("Cannot random search without e_c or i_c set")
        elif self.random:
            if self.random==1:
                random_ops = {'e_c': self.e_c[-2], 'i_c': self.i_c[-2]}
                prune = True
                num_patterns =  self.hypers['total_patterns']-1
            else:
                random_ops = {'e_c': self.e_c[-1], 'i_c': self.i_c[-1]}
                prune = self.random==2
                num_patterns = self.hypers['total_patterns']
            model_id = self.model_id + '_r{}'.format(self.random)
            jn_print("Generating model at random level {}, e_c={}, i_c={}, prune={}".format(
                    self.random,
                    random_ops['e_c'],
                    random_ops['i_c'],
                    prune))
        elif 'genotype' not in self.hypers:
            random_ops = None
            prune = True
            num_patterns = self.start_size
            model_id = self.model_id
        
        if 'genotype' in self.hypers:
            self.model = Net(
                dim=self.dim,
                dataset_name=self.hypers['dataset']['name'],
                lr_schedule=self.hypers['lr_schedule'],
                prune=self.hypers['prune'],
                genotype=self.hypers['genotype'])
            self.start_size = self.model.built_patterns
            self.hypers['total_patterns'] = self.model.built_patterns
        else:
            self.model = Net(
                dim=self.dim,
                classes=self.hypers['dataset']['classes'],
                dataset_name=self.hypers['dataset']['name'],
                scale=self.hypers['scale'],
                patterns=self.hypers['patterns'],
                num_patterns=num_patterns,
                total_patterns=self.hypers['total_patterns'],
                nodes=self.hypers['nodes'],
                depth=self.hypers['depth'],
                random_ops=random_ops,
                prune=prune,
                model_id = model_id,
                drop_prob=self.hypers['drop_prob'],
                lr_schedule=self.hypers['lr_schedule'])
        
        
        
        if self.random==1:
            self.model.add_pattern()
        
        self.model.data = self.data
        self.model.dataset_name = self.hypers['dataset']['name']

    def reinit(self):
        num_patterns = self.model.built_patterns
        self.start_size = num_patterns
        genotype = self.model.extract_genotype(weights=False)[1]
        self.model = Net(
            dim=self.dim,
            genotype=genotype,
            classes=self.hypers['dataset']['classes'],
            dataset_name=self.hypers['dataset']['name'],
            scale=self.hypers['scale'],
            patterns=self.hypers['patterns'],
            num_patterns=num_patterns,
            total_patterns=self.hypers['total_patterns'],
            nodes=self.hypers['nodes'],
            depth=self.hypers['depth'],
            random_ops=None,
            prune=self.hypers.get('prune', True),
            model_id = self.model_id,
            drop_prob=self.hypers['drop_prob'],
            lr_schedule=self.hypers['lr_schedule'])
        self.model.data = self.data
        
    def track_compression(self):
        _, e_c, i_c = self.model.genotype_compression()
        self.e_c.append(e_c)
        self.i_c.append(i_c)

    def reinit_train(self):
        start_t = time.time()
        curr_patterns = self.model.built_patterns
        while curr_patterns < self.hypers['total_patterns']:
            print("Built {} of {} patterns.".format(curr_patterns, self.hypers['total_patterns']))
            curr_patterns = self.train(verbose=False)
            self.reinit()
            
#         n_prune_loops = 2
#         for _ in range(n_prune_loops):
#             self.train(verbose=False)
#             if _ != n_prune_loops-1:
#                 self.reinit()
        
        print('Finished!')
        jn_print("Search Time: {}".format(show_time(time.time() - start_t)))
        self.hypers['lr_schedule']['T'] = 600
        #self.hypers['prune'] = False
        self.reinit()
        self.train(verbose=False)

    def train(self, verbose=True):
        if self.model is None:
            self.generate_model()

        if not self.random:
            # search
            search_start = time.time()
            for n in range(self.start_size, self.hypers['total_patterns']):
                jn_print(str(self.model))
                comp_ratio = self.sizes.get(n, 0)
                aim = comp_ratio * (.9 if self.sizes.get(n, 0) > .35 else .66)
                jn_print("=== {} Patterns. Target Comp: {:.2f}, Aim: {:.2f}".format(n, comp_ratio, aim))

                # learn+prune
                met_thresh = full_train(self.model,
                                        comp_lambdas=self.hypers['prune_rate'],
                                        comp_ratio=aim,
                                        size_thresh=self.sizes[n],
                                        nas_schedule=self.hypers['nas_schedule'])
                clean(verbose=False)
                if met_thresh:
                    if n != self.hypers['total_patterns']:
                        jn_print("Adding next pattern: {}".format(n + 1))
                        self.track_compression()
                        self.model.add_pattern()
                    return n+1
                if not self.model.lr_schedulers['init'].remaining:
                    return n

            # print search stats
            clean("Search End")
            if verbose:
                jn_print("Search Time: {}".format(show_time(time.time() - search_start)))
            if len(self.e_c):
                jn_print("Edge Comp: {} Input Comp: {}".format(self.e_c[-1], self.i_c[-1]))
            jn_print(str(self.model))
        else:
            jn_print(str(self.model))
            #self.model.detail_print()

        # train
        full_train(self.model,
                   epochs=self.model.lr_schedulers['init'].remaining,
                   nas_schedule=self.hypers['nas_schedule'])
        self.track_compression()
        clean()

    def random_search(self, level, e_c=None, i_c=None):
        self.random = level
        del self.model
        self.model = None
        if e_c is not None:
            self.e_c, self.i_c = [e_c]*2, [i_c]*2
        print(self.model)
        self.train()
