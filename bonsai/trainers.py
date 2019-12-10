import datetime
import time

import torch.nn as nn
import torch.optim as optim

from bonsai.helpers import *

# set up logging
setup_logger("training_logger", filename='logs/trainer.log', mode='a', terminator="\n")
setup_logger("jn_out", filename='logs/jn_out.log', mode='a', terminator="")
training_logger = logging.getLogger('training_logger')
jn_out = logging.getLogger('jn_out')

log_print = log_print_curry([training_logger, jn_out])
jn_print  = log_print_curry([jn_out])


# === EPOCH LEVEL FUNCTIONS ============================================================================================
def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    log_print("\n\x1b[31mAdjusting lr to {}\x1b[0m".format(lr))
    return lr


def extract_curr_lambdas(comp_lambdas,epoch):
    comp_lambda = None
    if comp_lambdas:
        if epoch>=comp_lambdas['transition']:
            comp_lambda = comp_lambdas['lambdas']
    return comp_lambda


# === CUSTOM LOSS FUNCTIONS ============================================================================================
def compression_loss(model, comp_lambda, comp_ratio, item_output=False):
    # edge pruning
    prune_sizes = []
    if comp_lambda['edge']>0:
        for cell in model.cells:
            prune_sizes += [torch.sum(torch.cat([pruner.mem_size*pruner.sg() for pruner in cell.edge_pruners])).view(-1)]
        edge_comp_ratio = torch.div(torch.cat(prune_sizes), model.edge_sizes)
        edge_comp = torch.norm(comp_ratio - edge_comp_ratio)
        edge_loss = comp_lambda['edge'] * edge_comp
    else:
        edge_loss = 0

    # input pruning
    if comp_lambda['input']>0:
        input_pruners = torch.cat(
            tuple(
                [torch.sum(torch.cat(tuple(pruner.sg() for pruner in cell.input_pruners))).view(-1) for cell in model.cells]
            ))
        input_comp_ratio = torch.div(input_pruners, model.input_p_tot)
        input_comp = torch.norm(1/model.input_p_tot - input_comp_ratio)
        input_loss = comp_lambda['input']*input_comp
    else:
        input_loss = 0

    loss = edge_loss+input_loss
    if item_output:
        ecr = 0 if edge_loss == 0 else torch.mean(edge_comp_ratio).item()
        icr = 0 if input_loss == 0 else torch.mean(input_pruners).item()
        return loss, [ecr,icr], [edge_loss,input_loss]
    else:
        return loss, None, None


# === PERFORMANCE METRICS ==============================================================================================
def top_k_accuracy(output, target, top_k, max_k):
    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].view(-1).float().sum(0) for k in top_k]


def accuracy_string(prefix, corrects, t_start, loader, top_k, comp_ratio=None, return_str=False):
    corrects = 100. * corrects / float(len(loader.dataset))
    out_string = "{} Corrects: ".format(prefix)
    for i, k in enumerate(top_k):
        out_string += 'Top-{}: {:.2f}%, '.format(k, corrects[i])
    if comp_ratio is not None:
        out_string += 'Comp: {:.2f}, {:.2f} '.format(*comp_ratio)
    out_string += show_time(time.time() - t_start)
    
    if return_str:
        return out_string
    else:
        log_print(out_string)


# === BASE LEVEL TRAIN AND TEST FUNCTIONS===============================================================================
def train(model, device, train_loader, **kwargs):
    # === tracking stats ======================
    top_k = kwargs.get('top_k', [1])
    max_k = max(top_k)
    corrects = np.zeros(len(top_k), dtype=float)
    comp_lambda = extract_curr_lambdas(kwargs.get('comp_lambdas'),kwargs['epoch'])

    # === train epoch =========================
    model.train()
    epoch_start = time.time()
    multiplier = kwargs.get('multiplier',1)
    jn_print(datetime.datetime.now().strftime("%m/%d/%Y %I:%M %p"))

    t_data_start = None
    t_cumul_data, t_cumul_ops = 0,0
    for batch_idx, (data, target) in enumerate(train_loader):
        t_data_end = time.time()
        if t_data_start is not None:
            t_cumul_data += (t_data_end-t_data_start)
        t_op_start = time.time()

        print_or_end = (not batch_idx % 10) or (batch_idx == len(train_loader)-1)
        batch_start = time.time()

        # pass data ===========================
        data, target = data.to(device), target.to(device)
        if (batch_idx % multiplier == 0) or (batch_idx == len(train_loader)-1):
            kwargs['optimizer'].zero_grad()

        verbose = kwargs['epoch'] == 0 and batch_idx == 0
        outputs = model.forward(data, model.drop_prob, auxiliary=True, verbose=verbose)

        # classification loss =================
        def loss_f(x): return kwargs['criterion'](x, target)
        losses = [loss_f(output) for output in outputs[:-1]]
        final_loss = loss_f(outputs[-1])
        loss = final_loss + .2 * sum(losses)

        # compression loss ====================
        comp_ratio = None
        if comp_lambda:
            comp_loss, comp_ratio, loss_components = compression_loss(model,
                                                                      comp_lambda=comp_lambda,
                                                                      comp_ratio=kwargs['comp_ratio'],
                                                                      item_output = print_or_end)
            if print_or_end:
                loss_components = [loss] + loss_components
            loss += comp_loss

        # end train step ======================
        loss = loss/multiplier
        loss.backward()
        if (batch_idx % multiplier == 0) or (batch_idx == len(train_loader) - 1):
            kwargs['optimizer'].step()
        corrects = corrects + top_k_accuracy(outputs[-1], target, top_k=kwargs.get('top_k', [1]), max_k=max_k)

        # mid epoch updates ===================
        if print_or_end:
            losses_out = [np.round(x.item(), 2) for x in losses] + [np.round(final_loss.item(),2)]
            losses_out = ", ".join(["{}: {}".format(i,x) for i,x in enumerate(losses_out)])
            prog_str = 'Train Epoch: {:<3} [{:<6}/{:<6} ({:.0f}%)]\t'.format(
                kwargs['epoch'],
                (batch_idx + 1) * len(data),
                len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader))
            if comp_lambda is not None:
                prog_str += 'Comp Ratio: [E: {:.3f}, I: {:.3f}]'.format(*comp_ratio)
                prog_str += ', Loss Comp: [C: {:.3f}, E: {:.3f}, I: {:.2f}], '.format(*loss_components)
            prog_str += 'Per Epoch: {:<7}, '.format(show_time((time.time() - batch_start) * len(train_loader)))
            prog_str += 'Alloc: {:<9}, '.format(mem_stats())
            prog_str += 'Data T: {:<6.3f}, Op T: {:<6.3f}'.format(t_cumul_data,t_cumul_ops)
            jn_print(prog_str, end="\r", flush=True)
        t_cumul_ops += (time.time()-t_op_start)
        t_data_start = time.time()

    # === output ===============
    jn_print(prog_str)
    accuracy_string("Train", corrects, epoch_start, train_loader, top_k, comp_ratio=comp_ratio)
    if comp_lambda is not None:
        log_print("Train Loss Components: C: {:.3f}, E: {:.3f}, I: {:.2f}".format(*loss_components))


def test(model, device, test_loader, top_k=[1]):
    # === tracking stats =====================
    max_k = max(top_k)
    corrects, e_corrects = np.zeros(len(top_k)), np.zeros(len(top_k))
    t_start = time.time()

    # === test epoch =========================
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            outputs = model.forward(data, drop_prob=0, auxiliary=True)
            #e_output = torch.mean(torch.stack(outputs, 1), 1)
            corrects = corrects + top_k_accuracy(outputs[-1], target, top_k=top_k, max_k=max_k)
            #e_corrects = e_corrects + top_k_accuracy(e_output, target, top_k=top_k, max_k=max_k)

    # === format results =====================
    #log_print(accuracy_string("All Towers Test ", e_corrects, t_start, test_loader, top_k, return_str=True))
    return accuracy_string("Last Tower Test ", corrects, t_start, test_loader, top_k, return_str=True)


def size_test(model, verbose=False):
    # run a few batches through the model to get an estimate of its GPU size
    try:
        start_size = mem_stats(False)/(1024**3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for batch_idx, (data, target) in enumerate(model.data[0]):
            data, target = data.to(device), target.to(device)
            out = model.forward(data, auxiliary=True, verbose=(verbose and batch_idx==0))
            loss = criterion(out[-1], target)
            loss.backward()
            if batch_idx > 2:
                break
        overflow = False
        size = mem_stats(False)/(1024**3)-start_size
        clean(verbose=False)
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            overflow = True
            model = model.to(torch.device('cpu'))
            size = mem_stats(False)/(1024**3)
            clean(verbose=False)
            del model
            try:
                del data, target
            except:
                pass
            clean(verbose=False)
        else:
            raise e
    return size, overflow


def sp_size_test(n, e_c, prune=True,**kwargs):
    with open("pickles/size_test_in.pkl","wb") as f:
        pkl.dump([n,e_c,prune,kwargs],f)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sizer.py')
    python = get_python3()
    s=subprocess.check_output([python, path])
    if kwargs.get('print_model',False):
        print(s.decode('utf8'))
    with open("pickles/size_test_out.pkl","rb") as f:
        return pkl.load(f)
    

# === FULL TRAINING HANDLER=============================================================================================
def full_train(model, epochs, **kwargs):
    # === unpack data/ tracking stats=========
    train_loader, test_loader = model.data

    # === learning handlers ==================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.SGD(model.parameters(), lr=model.lr_scheduler.lr, momentum=.9, weight_decay=3e-4)
    model.jn_print, model.log_print = jn_print, log_print
    print("=== Training {} ===".format(model.model_id))

    # === init logging =======================
    training_logger.info("=== NEW FULL TRAIN ===")
    log_print("Starting at {}".format(datetime.datetime.now()))
    training_logger.info(model.creation_string())

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    # === run n epochs =======================
    for epoch in range(model.lr_scheduler.t, model.lr_scheduler.t+epochs):
        training_logger.info("=== EPOCH {} ===".format(epoch))

        # train =========================
        train(model, device, train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch, **kwargs)
        model.save_genotype()

        # prune ==============================
        comp_lambda = extract_curr_lambdas(kwargs.get('comp_lambdas'),epoch)
        if comp_lambda != None:
            model.eval()
            [pruner.track_gates() for pruner in model.edge_pruners + model.input_pruners]
            if epoch and not (epoch + 1) % kwargs['prune_interval']:
                model.deadhead()

        if model.prune:
            hard, soft = model.genotype_compression()[:2]
            if soft and hard:
                jn_print("Soft Comp: {:.3f}, Hard Comp: {:.3f}".format(soft, hard))

        # test ===============================
        log_print(test(model, device, test_loader, top_k=kwargs.get('top_k', [1])))

        # anneal =============================
        set_lr(optimizer, model.lr_scheduler.step())

    return True
