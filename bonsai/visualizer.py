import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
import numpy as np
import re
import sys

plt.style.reload_library()
plt.style.use('material')


# === HELPERS ==========================================================================================================
def time_parse(t):
    if "m" in t:
        return int(t.split("m")[0]) * 60 + int(t.split(",")[1].split('s')[0])
    else:
        return float(t.split('s')[0])


# === DATA LOADING =====================================================================================================
path = ""


def get_prune_logs():
    with open(path+"logs/jn_out.log", "r") as f:
        out = f.read().replace('\x00', '')
    return out.split("\n")


def get_raw_runs():
    with open(path+"logs/trainer.log") as f:
        data = f.read()
    runs = data.split('=== NEW FULL TRAIN ===')
    return [run for run in runs if 'Starting at' in run]


# === PRUNE VISUALIZATION ==============================================================================================
def scrape(prog=False):
    rows = []
    row = {}
    aim, target = 0, 0
    logs = get_prune_logs()
    curr_prog = int([log for log in logs if '%' in log and 'Train Epoch' in log][-1].split("(")[1].split("%")[0])
    if prog:
        return curr_prog

    for line in logs:
        if line == "" and 'Test AT Acc' in row:
            row['Aim Comp'] = aim
            row['Target Comp'] = target
            if row:
                rows.append(row)
            row = {}
        if 'Adjusting lr' in line:
            row['Learning Rate'] = float(line.split("to")[1].split("\x1b[0m")[0])
        if 'Target Comp' in line:
            target = float(line.split(":")[1].split(",")[0])
            if 'Aim' in line:
                aim = float(line.split(":")[-1])
        if 'Train Epoch' in line:
            if 'Loss:' in line:
                row['C Loss'] = float(line.split(":")[2].split(",")[0])
            row['Alloc'] = float(line.split(":")[-1].split("GiB")[0].replace("|carr_ret|", ""))
        if 'Train Corrects' in line:
            row['Train Acc'] = float(line.split(":")[2].split(",")[0][:-1])
            if 'Comp' in line:
                row['Edge Comp'] = float(line.split(":")[3].split(",")[0])
                row['Input Comp'] = float(line.split(":")[3].split(" ")[2])
            row['Runtime'] = time_parse(line.split(" ")[-1].strip())
        if 'Hard Comp' in line:
            row['Soft Comp'] = float(line.split(':')[1].split(",")[0].strip())
            row['Hard Comp'] = float(line.split(':')[2].strip())
        if 'Train Loss Components' in line:
            row['C Loss'] = float(line.split(":")[2].split(",")[0])
            row['E Loss'] = float(line.split(":")[3].split(",")[0])
            row['I Loss'] = float(line.split(":")[4].split(",")[0])
        if 'Test' in line and 'Corrects' in line:
            if 'All Towers' in line:
                row['Test AT Acc'] = float(line.split(":")[2].split("%")[0])
            elif 'Last Tower' in line:
                row['Test LT Acc'] = float(line.split(":")[2].split("%")[0])
    return pd.DataFrame(rows), curr_prog


class PruneAnimator:
    def __init__(self, axes, col_sets):
        self.axes = axes
        self.col_sets = col_sets
        self.prog = 0

    def animate(self, i):
        df, prog = scrape()
        cycles = []
        if 'E_Loss' in list(df):
            e_losses = df[df['E_Loss'].isnull()].index
            cycles = [x for i, x in enumerate(e_losses) if x != e_losses[i - 1] + 1]

        for i, cols in enumerate(self.col_sets):
            dmin, dmax = 1000, 0
            self.axes[i].clear()
            labeled = True
            for label in cols:
                if label not in list(df):
                    labeled=False
                    continue
                self.axes[i].plot(df[label], label=label)
                if df[label].min(skipna=True) < dmin:
                    dmin = df[label].min(skipna=True)
                if df[label].max(skipna=True) > dmax:
                    dmax = df[label].max(skipna=True)
            for cycle in cycles:
                self.axes[i].plot([cycle] * 100, np.linspace(dmin, dmax, 100), c='k', alpha=.25)
            if labeled:
                self.axes[i].legend(fontsize=7)
            if any(['Acc' in c for c in cols]):

                self.axes[i].set_yticks(np.arange(np.floor(dmin/10)*10, np.ceil(dmax/10)*10,10))
            self.axes[i].set_title(", ".join(cols), fontsize=7)

        if prog != self.prog:
            self.axes[-1].clear()
            self.axes[-1].barh(1, prog, align='center', color='#FFCB6B')
            self.axes[-1].set_xlim(0, 100)
            self.axes[-1].set_title("Epoch {} Progress".format(len(df)), fontsize=7)
            self.axes[-1].set_yticks([])
            self.axes[-1].set_xticks(range(0, 110, 10))
            self.prog = prog
        return self.axes


def plot_prune(figsize=(10, 4)):
    col_sets = [
        ['Train Acc', 'Test LT Acc', 'Test AT Acc'],
        ['C Loss', 'E Loss', 'I Loss'],
        ['Hard Comp', 'Soft Comp', 'Aim Comp', 'Target Comp'],
        ['Input Comp']
    ]
    fig = plt.figure(figsize=figsize, dpi=125)
    gs = gridspec.GridSpec(len(col_sets) + 1, 1, height_ratios=[1] * len(col_sets) + [.25])
    axes = [plt.subplot(g) for g in gs]
    plt.subplots_adjust(left=.05, right=.95, top=.95, bottom=.05, hspace=.35)
    animator = PruneAnimator(axes, col_sets)
    ani = animation.FuncAnimation(fig, animator.animate, interval=1000)
    plt.show()


# === TRAIN SCRAPING ===================================================================================================
def accuracy(raw_line, prefix, run_details):
    top1 = "{} Top-1".format(prefix)

    if run_details.get(top1) is None:
        run_details[top1] = []
    if 'Top-1' in raw_line:
        run_details[top1].append(float(raw_line.split(':', 2)[-1].split("%")[0]))
    if 'Top' not in raw_line:
        run_details[top1].append(float(raw_line.split(':', 1)[-1].split("%")[0]))
    return run_details


def proc_run(run):
    run_details = {}
    curr_epoch = -1
    deadhead_history = []
    param_history = []
    epochs = []
    a_loss, e_loss, i_loss = [], [], []

    prev_line = ""
    for raw_line in run.split("\n"):
        if raw_line == prev_line:
            prev_line = raw_line
            continue

        # add new values to histories
        if 'EPOCH' in raw_line:
            deadhead_history.append(0)
            param_history.append(0)
            epochs.append(int(raw_line.split("EPOCH")[1].split(" ===")[0]))

        # track model stats
        if 'Starting at' in raw_line:
            run_details['Start Time'] = raw_line.split("Starting at")[1]
        if 'Dim' in raw_line:
            if 'torch' in raw_line:
                raw_line = raw_line.replace(")", "").replace("torch.Size(", "")
            detail_str = re.split(',(?=\s[A-Za-z])', raw_line)
            new_str = ''
            for detail in detail_str:
                try:
                    k, v = detail.split(":")
                except Exception as e:
                    pass
                new_str += "'" + k.strip() + "':" + v
                if detail != detail_str[-1]:
                    new_str += ", "
            locals_ = locals()
            if '<' in new_str and '>' in new_str:
                new_str = new_str.split("<")[0] + 'None' + new_str.split(">")[-1]
            exec('details={' + new_str + "}", None, locals_)
            run_details.update(locals_['details'])

        # add accuracies
        if 'Train Corrects:' in raw_line:
            run_details = accuracy(raw_line, 'Train', run_details)
        elif ('Last Towers Test' in raw_line and 'Corrects' in raw_line) or (
                'Test' in raw_line and 'Towers' not in raw_line and 'Corrects' in raw_line) or 'test acc' in raw_line:
            run_details = accuracy(raw_line, 'LT Test', run_details)
        elif ('All Towers Test' in raw_line and 'Corrects' in raw_line):
            run_details = accuracy(raw_line, 'AT Test', run_details)

        # track deadheading
        if 'Deadheaded' in raw_line:
            deadhead_history[-1] = -int(raw_line.split('Deadheaded')[-1].split("operations")[0])
        if 'Param Delta' in raw_line:
            param_history[-1] = [int(raw_line.split('Param Delta:')[-1].split("->")[0].replace(",", "")),
                                 int(raw_line.split('->')[-1].replace(",", ""))]

        # track loss_comps
        if 'Train Loss Components' in raw_line:
            loss_comps = raw_line.split(':')
            a, e, i = [float(loss.split(",")[0]) for loss in loss_comps[2:]]
            a_loss, e_loss, i_loss = a_loss + [a], e_loss + [e], i_loss + [i]
        if raw_line != "":
            prev_line = raw_line
    run_details['Loss Accuracy'] = a_loss
    run_details['Loss Edge'] = e_loss
    run_details['Loss Input'] = i_loss
    run_details['Epochs'] = epochs

    # flatten param hist
    if [x for x in param_history if type(x) is not int]:
        prev_val = [hist[0] for hist in param_history if hist != 0][0]
        new_param_hist = []
        for val in param_history:
            if val == 0:
                new_param_hist.append(prev_val)
            else:
                new_param_hist.append(val[1])
                prev_val = val[1]
        run_details['Params'] = new_param_hist
    run_details['Deadhead'] = deadhead_history
    return run_details


def proc_all_runs():
    runs = [proc_run(run) for run in get_raw_runs()]
    runs = pd.DataFrame(runs)
    for col in [col for col in list(runs) if 'Top' in col]:
        runs[col] = runs[col].apply(lambda x: x if type(x) == list else [])
    runs['LT Test Top-1 Max'] = runs['LT Test Top-1'].apply(lambda x: max(x, default=0))
    runs['AT Test Top-1 Max'] = runs['AT Test Top-1'].apply(lambda x: max(x, default=0))
    runs['Epoch'] = runs['Epochs'].apply(lambda x: x[-1] if len(x) else 0)
    return runs


# === TRAIN VISUALIZATION ==============================================================================================
class TrainAnimator:
    def __init__(self, axes):
        self.axes = axes
        self.prog = 0
        self.curr_epoch = 0

    def animate(self, i):
        runs = proc_all_runs()
        full_runs = runs[runs[['LT Test Top-1','Epochs']].\
            apply(lambda x: len(x['LT Test Top-1']) > 512,axis=1)]. \
            sort_values(by='LT Test Top-1 Max', ascending=False)


        compare = full_runs['LT Test Top-1'].values[0]
        compare_str = 'PR'

        at, lt = max(runs.iloc[-1]['AT Test Top-1']), max(runs.iloc[-1]['LT Test Top-1'])
        at_last, lt_last = runs.iloc[-1]['AT Test Top-1'][-1], runs.iloc[-1]['LT Test Top-1'][-1]
        curr_run = runs.iloc[-1]
        epoch = runs.iloc[-1]['Epoch']
        cm = plt.cm.Spectral

        # plot previous runs
        if epoch!=self.curr_epoch:
            self.axes[0].clear()
            self.axes[1].clear()
            for i, (idx,run) in enumerate(full_runs.iterrows()):
                ys = run['LT Test Top-1']
                xs = list(run['Epochs'])[:len(ys)]
                self.axes[0].plot(xs,ys, color=cm(i / len(full_runs)), alpha=.75 if i == 0 else .5)

        # plot current run
        if epoch < 0:
            print("No log yet...")
        else:
            if epoch!=self.curr_epoch:
                print()
                curr = curr_run['LT Test Top-1'][-1]
                curr_max = max(curr_run['LT Test Top-1'])
                curr_arg_max = np.argmax(curr_run['LT Test Top-1'])
                rec, rec_max, rec_arg_max = compare[epoch], max(compare[:epoch + 1]), np.argmax(compare[:epoch + 1])

                text = "==== EPOCH {} ======================================\n".format(epoch)
                text += "AT Max: {} LT Max: {}\n".format(at, lt)
                text += "AT Last: {} LT Last: {}\n".format(at_last, lt_last)
                text += "Current Delta to {}:     {:> 2.2f}% ({}% vs {}%)\n".format(compare_str, curr - rec, curr, rec)
                text += "Current Delta to {} Max: {:> 2.2f}% ({}% @{} vs {}% @{})".format(compare_str,
                                                                                          curr_max - rec_max,
                                                                                          curr_max,
                                                                                          curr_arg_max,
                                                                                          rec_max,
                                                                                          rec_arg_max)
                self.axes[0].text(-30,100.5,text,fontsize=8, fontfamily='monospace')
                ys = curr_run['LT Test Top-1']
                xs = list(curr_run['Epochs'])[:len(ys)]
                self.axes[0].plot(xs,ys, color='k', linewidth=1.5)
                self.axes[0].set_ylim(min(curr_run['LT Test Top-1'][-10:])-1, 100)
                self.axes[0].set_title("CIFAR-10 Loss History, Bonsai Net")
                self.axes[0].set_xlabel("Epoch", fontsize=8)
                self.axes[0].set_ylabel("Accuracy", fontsize=8)
                self.axes[0].set_yticks(np.arange(int(min(curr_run['LT Test Top-1'][-10:])), 100, 1))
                self.axes[0].tick_params(axis='both', which='major', labelsize=8)
                self.curr_epoch = epoch

        prog = scrape(prog=True)
        if prog != self.prog:
            self.axes[-1].clear()
            self.axes[-1].barh(1, prog, align='center', color='#FFCB6B')
            self.axes[-1].set_xlim(0, 100)
            self.axes[-1].set_title("Epoch Progress", fontsize=7)
            self.axes[-1].set_yticks([])
            self.axes[-1].set_xticks(range(0, 110, 10))
            self.prog = prog
        return self.axes


def plot_train(figsize=(10, 4)):
    fig = plt.figure(figsize=figsize, dpi=125)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1/10])
    axes = [plt.subplot(g) for g in gs]

    plt.subplots_adjust(left=.05, right=.95, top=.89, bottom=.05, hspace=.15)
    animator = TrainAnimator(axes)
    ani = animation.FuncAnimation(fig, animator.animate, interval=1000)
    plt.show()


# === MAIN =============================================================================================================
if __name__=='__main__':
    path = "../"
    if sys.argv[1]=='p':
        plot_prune()
    else:
        plot_train()