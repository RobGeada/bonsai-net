from bonsai.data_loaders import load_data
from bonsai.net_vert import Net
from bonsai.trainers import size_test

import pickle as pkl

if __name__=='__main__':
        with open("size_test_in.pkl","rb") as f:
            [n,e_c,kwargs]=pkl.load(f)
    
        data, data_shape = load_data(kwargs['batch_size'], kwargs['dataset'])
        model = Net(dim=data_shape,
                    classes=kwargs['classes'],
                    scale=kwargs['scale'],
                    patterns=kwargs['patterns'],
                    num_patterns=n,
                    random_ops={'e_c': e_c, 'i_c': 1., 'worst_case': True},
                    nodes=kwargs['nodes'],
                    prune=True,
                    auxiliary=True)
        if not kwargs.get('raw', False):
            model.add_pattern(full_ops=True)
        if kwargs.get('detail', False):
            model.detail_print()
        if kwargs.get('print_model', False):
            print(model)
        out = size_test(model, data, half=kwargs.get('half', False), verbose=kwargs.get('verbose',False))
        with open("size_test_out.pkl","wb") as f:
            pkl.dump(out,f)