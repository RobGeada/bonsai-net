from bonsai.ops import commons, PrunableOperation
from bonsai.helpers import mem_stats, clean
import sys
import torch
import pprint

# read args
C = int(sys.argv[2])
dim = [int(x) for x in sys.argv[1:]]

# build op
op_mems = {}
input = torch.zeros(dim).cuda()

for op, f in commons.items():
    start_mem = mem_stats(False)
    op_f = f(C, 1).cuda()
    out = op_f(input)
    end_mem = mem_stats(False)-start_mem
    del out, op_f
    clean(verbose=False)
    op_mems[op]=end_mem/1024/1024
pp = pprint.PrettyPrinter(indent=0)
pp.pprint(op_mems)




