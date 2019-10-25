import gc
import logging
import math
import numpy as np
import signal
import torch


# === TORCH HELPERS ====================================================================================================
def clean(name=None, verbose=True):
    pre = mem_stats()
    gc.collect()
    torch.cuda.empty_cache()
    if verbose:
        print('Cleaning at {}. Pre: {}, Post: {}'.format(name, pre, mem_stats()))


def t_extract(x):
    return np.round(x.item(), 2)


def pretty_size(size):
    # pretty prints a torch.Size object
    assert (isinstance(size, torch.Size))
    return " x ".join(map(str, size)), int(np.prod(list(map(int, size))))


# Code by James Bradbury - https://github.com/jekbradbury
def print_obj_tree(min_elements=None):
    obj_list = [obj for obj in gc.get_objects() if torch.is_tensor(obj) or isinstance(obj, torch.autograd.Variable)]
    for obj in obj_list:
        if min_elements and obj.nelement() < min_elements: continue
        referrers = [r for r in gc.get_referrers(obj) if r is not obj_list]
        print(f'{id(obj)} {obj.__class__.__qualname__} of size {tuple(obj.size())} with references held by:')
        for referrer in referrers:
            if torch.is_tensor(referrer) or isinstance(referrer, torch.autograd.Variable):
                info_str = f' of size {tuple(referrer.size())}'
            elif isinstance(referrer, dict):
                info_str = ' in which its key is ', [k for k, v in referrer.items() if v is obj]
            else:
                info_str = ''
            print(f'  {id(referrer)} {referrer.__class__.__qualname__}{info_str}')


# === MODEL HELPERS ====================================================================================================
def schedule_generator(lr):
    return lambda x: {'lr_min': lr['lr_min'], 'lr_max': lr['lr_max'], 't_0': x, 't_mult': 1}


def general_num_params(model):
    # return number of differential parameters of input model
    return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])


def channel_mod(dim, to):
    return dim[0], to, dim[2], dim[3]


def width_mod(dim, by):
    return dim[0], dim[1], dim[2] // by, dim[3] // by


def batch_mod(dim, by):
    return dim[0] // by, dim[1], dim[2], dim[3]


def cw_mod(dim, by):
    return dim[0], dim[1] * by, dim[2] // by, dim[3] // by

# === I/O HELPERS ======================================================================================================
class BST:
    def __init__(self, lower, upper, depth=6):
        self.lower = lower
        self.upper = upper
        self.step = (upper - lower) / 2 / 2
        self.pos = self.lower + (upper - lower) / 2
        self.depth = 0
        self.max_depth = depth
        self.min_step = (self.upper - self.lower) / (2 ** (self.max_depth - 1))
        self.answer = None
        self.passes = []
        self.pass_dict={}

    def query(self, result, g_comp):
        if self.pos <= self.lower:
            self.answer = self.lower
        elif self.pos >= self.upper:
            self.answer = self.upper
        elif self.depth == self.max_depth:
            self.answer = self.pos

        if result:
            self.pos -= self.step
            if self.step > self.min_step:
                self.step /= 2
            self.depth += 1
        else:
            self.passes.append(self.pos)
            self.pass_dict[self.pos]=g_comp
            self.pos += self.step
            if self.step > self.min_step:
                self.step /= 2
            self.depth += 1


def looping_generator(l):
    n = 0
    while 1:
        yield l[n % len(l)]
        n += 1


class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.signals = 0
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        self.signals += 1
        if self.signals > 1:
            exit(0)
        print("\nDelaying interrupt for epoch end. Interrupt again to force kill.".format(self.signals))

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


class TransitionDict:
    def __init__(self, d={}):
        self.d = d
        self.keys = list(d.keys())

    def __getitem__(self, index):
        if not len(self.keys):
            return None
        for i, key in enumerate(self.keys):
            if i == len(self.keys) - 1 and self.keys[-1] <= index:
                return self.d[self.keys[-1]]
            elif self.keys[i] <= index < self.keys[i + 1]:
                return self.d[self.keys[i]]

    def __ne__(self, other):
        if other is None and len(self.keys):
            return True
        elif other is None and not len(self.keys):
            return False
        else:
            raise NotImplementedError


def div_remainder(n, interval):
    # finds divisor and remainder given some n/interval
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder


def namer():
    # generate random tripled-barrelled name to track models
    names = open("bonsai/names.txt", "r").readlines()
    len_names = len(names)
    choices = np.random.randint(0, len_names, 3)
    return " ".join([names[i].strip() for i in choices]).replace("'", "")


def sizeof_fmt(num, suffix='B'):
    # turns bytes object into human readable
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.2f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.2f%s%s" % (num, 'Yi', suffix)


def mem_stats(human_readable=True):
    # returns current allocated torch memory
    if human_readable:
        return sizeof_fmt(torch.cuda.memory_cached())
    else:
        return int(torch.cuda.memory_cached())


def show_time(seconds):
    # show amount of time as human readable
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)


def setup_logger(logger_name, filename, mode, terminator):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter()
    file_handler = logging.FileHandler(filename, mode=mode)
    file_handler.setFormatter(formatter)
    # file_handler.terminator = terminator

    l.setLevel(logging.INFO)
    l.addHandler(file_handler)


def log_print_curry(loggers):
    def log_print(string, end='\n', flush=False):
        print(string, end=end, flush=flush)
        for logger in loggers:
            if end == "\r":
                logger.info(string + "|carr_ret|")
            else:
                logger.info(string)

    return log_print


def prev_output(raw=False):
    with open("logs/jn_out.log", "r") as f:
        if raw:
            print(repr(f.read()))
        else:
            print(f.read().replace("|carr_ret|\n", "\r"))


def wipe_output():
    open("logs/jn_out.log", "w").close()
