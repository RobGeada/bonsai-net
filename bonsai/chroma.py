from datetime import datetime
from IPython.display import clear_output
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import textwrap
import time

from bonsai.ops import commons

plt.style.use('material')


# === COLORS ===========================================================================================================
def hex_parse(h):
    r = int(h[1:3], 16)
    g = int(h[3:5], 16)
    b = int(h[5:7], 16)
    a = int(h[7:9], 16)
    return r, g, b, a


def color_create():
    hexa = str(hex(50))[2:]
    colors = {
        'Identity': {'hex': '#E53935ff'},
        'Avg_Pool_3x3': {'hex': '#5050ffff'},
        'Max_Pool_3x3': {'hex': '#0000ffff'},
        'Max_Pool_5x5': {'hex': '#0000cfff'},
        'Max_Pool_7x7': {'hex': '#00009fff'},
        'Conv_1x1': {'hex': '#fff000ff'},
        'Conv_3x3': {'hex': '#ffd000ff'},
        'Conv_5x5': {'hex': '#ffb000ff'},
        'Conv_7x7': {'hex': '#ff9000ff'},
        'Sep_Conv_3x3': {'hex': '#961696ff'},
        'Sep_Conv_5x5': {'hex': '#761676ff'},
        'Sep_Conv_7x7': {'hex': '#561656ff'},
        'Dil_Conv_3x3': {'hex': '#009a46ff'},
        'Dil_Conv_5x5': {'hex': '#007a26ff'},
        'Dil_Conv_7x7': {'hex': '#005a06ff'},
    }
    colors = {op: v for op, v in colors.items() if op in commons.keys()}
    colors = {op: {'hex': colors[op]['hex'], 'pos': i} for i, op in enumerate(colors.keys())}

    for color in colors.values():
        color['hexa'] = color['hex'][:-2] + hexa
        color['rgb'] = hex_parse(color['hex'])
        color['rgba'] = hex_parse(color['hexa'])
    return colors


# === HELPERS ==========================================================================================================
def node_remap(node):
    if node == 'x':
        return 0
    elif node == 'y':
        return 1
    else:
        return int(node) + 2


# === GENOTYPE PROCESSING ==============================================================================================
def process(name=None, genotype=None, params=None, g_viz=True):
    colors = color_create()
    if genotype is None:
        params, genotype = pkl.load(open('genotypes/{}_np.pkl'.format(name), "rb"))

    # create graph
    if g_viz:
        p = graphviz.Digraph(name='parent')
        p.attr('node', shape='circle', size='1,2')

    # get list of cell types from patterns
    cell_types, cell_positions, loop_idx = [], [], 0
    while len(cell_types) < len(genotype.items()):
        for i, pattern in enumerate(params['patterns']):
            for j, cell in enumerate(pattern):
                cell_positions.append([i + (loop_idx * len(params['patterns'])), j])
                cell_types.append(cell.upper()  if i else "N")
        loop_idx += 1

    # iterate through cells and get cell info
    cells = []
    for i, (cell_name, cell_genotype) in enumerate(genotype.items()):

        # parse intercell connectivity
        if cell_genotype['Y'].get('weights'):
            ys = [x.item() for x in cell_genotype['Y']['weights']]
        else:
            ys = [-1 if x else 1 for x in cell_genotype['Y']['zeros']]
        cell = {'name': 'Cell {} ({})'.format(i, cell_types[i]),
                'pos': cell_positions[i],
                'X': i - 1 if i else 'In',
                'Y': [idx - 1 if idx else 'In' for idx, val in enumerate(ys) if val > 0]}
        if g_viz:
            g = graphviz.Digraph()
            g.attr(label='{} ({})'.format(cell_name, cell_types[i]))
            g.attr('node', shape='circle')
            g.edge('{}_out'.format(i - 1) if i else 'In', '{}_x'.format(i))
            for idx, val in enumerate(ys):
                if val > 0:
                    g.edge('{}_out'.format(idx - 1) if idx else "In", '{}_y'.format(i), color='#000000ff')

        # parse intracell connectivity
        for key, val in cell_genotype.items():
            for v in val:
                if type(v) is not str:
                    weight = v[1] if type(v[1]) is int else v[1].item()
                    if key is not 'X' and key is not 'Y':
                        # parse connection origin and target
                        origin_node, target_node = key.split("->")
                        origin = '{}_{}'.format(cell_name, origin_node)
                        o, t = node_remap(origin_node), node_remap(target_node)
                        target = '{}_{}'.format(cell_name, target_node if t != params['nodes'] + 1 else 'out')

                        # add node dicts to cell if not present
                        if cell.get(o) is None:
                            cell[o] = {}
                        if cell[o].get(t) is None:
                            cell[o][t] = [None] * len(colors.keys())

                        # remap reduction cell initial identities
                        if cell_types[i] == 'R' and o <= 1 and v[0] == 'Identity':
                            op = 'Max_Pool_3x3'
                            op_pos = colors['Identity']['pos']
                        else:
                            op = v[0]
                            op_pos = colors[op]['pos']

                        # map unpruned operations
                        if weight > 0:
                            cell[o][t][op_pos] = colors[op]['rgb']
                            if g_viz:
                                g.edge(origin, target, color=colors[op]['hex'])
                        # map pruned operations
                        elif weight <= 0:
                            cell[o][t][op_pos] = colors[op]['rgba']
                            if g_viz:
                                g.edge(origin, target, color=colors[op]['hex'])
        cells.append(cell)

        # render cell
        if g_viz:
            g.render('graphs/graph-output-{}.gv'.format(i), view=False)
            p.subgraph(g)

    # render whole graph
    if g_viz:
        p.render('graph-output.gv', view=False)
    return params, cells, params['nodes'] + 2


# === PLOTTING FUNCTIONS ===============================================================================================
def plot_pixel(pixel, size, mod):
    out = np.zeros([size, size, 4])
    for i in range(size):
        for j in range(size):
            if pixel[i // mod] is not None:
                if j < size / 2 or pixel[i // mod][-1] == 255:
                    out[j][i] = pixel[i // mod][:-1] + (255,)
    return out


def plot_cell(cell, n, ax, padding=1, mod=2):
    size = len(commons) * mod
    g_size = size + 2 * padding
    out = np.zeros((n * g_size, n * g_size, 4), dtype=int)

    ax.axis('off')
    if cell is not None:
        ax.axis('on')
        # fill axis with white
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = (255, 255, 255, 255)
        # plot operations in cell
        for i, v in cell.items():
            if i not in ['name', 'pos', 'X', 'Y']:
                for j in v.keys():
                    if j > i and not (i == 0 and j == 1):
                        out[(i * g_size) + padding:(i * g_size + g_size) - padding,
                        (j * g_size) + padding:(j * g_size + g_size) - padding] = plot_pixel(cell[i][j], size, mod)

    # plot cell
    ax.imshow(out)

    # write cell connections
    fontsize = 12
    if cell is not None:
        x_str = str(cell['X']).replace("'", "").replace("[", '').replace("]", '')
        y_str = str(cell['Y']).replace("'", "").replace("[", '').replace("]", '')
        text_str = "X: {}\n\nY: {}".format(textwrap.fill(x_str, 20), textwrap.fill(y_str, 20))
        ax.text(0.05, 0.95, text_str, fontsize=fontsize, color='k', verticalalignment='top')
        ax.set_title(cell['name'], fontsize=fontsize * 1.2)

    # Major ticks
    ax.set_xticks(np.arange(size / 2 - .5, n * g_size, g_size))
    ax.set_yticks(np.arange(size / 2 - .5, n * g_size, g_size))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(0, n, 1), fontsize=fontsize)
    ax.set_yticklabels(np.arange(0, n, 1), fontsize=fontsize)

    # Minor ticks
    ax.set_xticks(np.arange(-.5, n * g_size, g_size), minor=True)
    ax.set_yticks(np.arange(-.5, n * g_size, g_size), minor=True)

    # grid lines based on minor ticks
    ax.grid(which='major', b=False)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)


# === MAIN =============================================================================================================
def plot(color_bar=False, g_viz=False):
    path = os.getcwd() + "/genotypes"
    target = max([(file, os.stat(path + "/" + file).st_mtime) for file in os.listdir(path)], key=lambda x: x[1])[0]
    target = target.replace("_np.pkl", "").replace(".pkl", "")
    idx = 0
    while 1:
        file_mod = os.stat('genotypes/{}.pkl'.format(target)).st_mtime
        colors = color_create()
        clear_output()
        try:
            print("Updated at", datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
            params, cells, n = process(target, g_viz=g_viz)

            # print operation color legend
            plt.figure(figsize=(16, .5), dpi=200)
            s = 10
            for i, (op, color) in enumerate(colors.items()):
                plt.fill_between([s * i, s * i + s], [0, 0], [s / 2, s / 2], color=color['hex'], linewidth=1)
                plt.text(s * i + s / 2,
                         s / 4, op,
                         color='white',
                         fontsize=9,
                         verticalalignment='center',
                         horizontalalignment='center')
            plt.xlim(0, s * len(colors))
            plt.axis('off')
            plt.show()
            if color_bar:
                return None

            # plot subplots
            dim = max([cell['pos'][0] for cell in cells]), max([cell['pos'][1] for cell in cells])
            fig, axes = plt.subplots(dim[0] + 1,
                                     dim[1] + 1,
                                     figsize=(16, 9 * (dim[0] + 1)),
                                     dpi=200)

            # plot cells in each subplot
            if dim[0]>0 and dim[1]>0:
                for a in [col for row in axes for col in row]:
                    plot_cell(None, n, a)
                for i, cell in enumerate(cells):
                    plot_cell(cell, n, axes[cell['pos'][0], cell['pos'][1]])
            elif dim[0]==0 or dim[1]==0:
                for a in axes:
                    plot_cell(None, n, a)
                for i, cell in enumerate(cells):
                    plot_cell(cell, n, axes[cell['pos'][1]])
            else:
                plot_cell(None, n, axes)
                for i, cell in enumerate(cells):
                    plot_cell(cell, n, axes)
            plt.tight_layout()
            plt.savefig("chroma_ims/{:03d}.png".format(idx))
            idx+=1
            plt.show()

        # catch file errors if no genotype is created yet
        except (EOFError, pkl.UnpicklingError):
            time.sleep(1)

        while file_mod == os.stat('genotypes/{}.pkl'.format(target)).st_mtime:
            time.sleep(1)
