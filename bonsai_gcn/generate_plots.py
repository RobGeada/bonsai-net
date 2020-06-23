# %%
import os, math
from io import BytesIO
import numpy as np
import pandas as pd
from plotnine import *  # ggplot, geom_point, aes, stat_smooth, facet_wrap, theme
from PIL import Image

output_dir = '../output'
mem_files = {'gae': '1795555_gae_mem.csv', 'gcn': '1795556_gcn_mem.csv'}
mem_plot = 'mem_plot.png'
gae_acc_file = 'gae_acc.csv'
gae_acc_plot = 'gae_acc_plot.png'
gcn_acc_file = 'gcn_acc.csv'
gcn_acc_plot = 'gcn_acc_plot.png'

#%%
###################################### Helper funcs
def get_concat(im1, im2, direction):
    assert direction == 'h' or direction == 'v', "Direction must be 'h' or 'v'"
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    if (direction == 'v'):
        dst.paste(im2, (0, im1.height))
    else:
        dst.paste(im2, (im1.width, 0))
    return dst
    
def stack_save_plots(plt1, plt2, output_file, direction):
    plt1_bytes = BytesIO()
    plt2_bytes = BytesIO()
    plt1.save(plt1_bytes)
    plt2.save(plt2_bytes)
    plt1_img = Image.open(plt1_bytes)
    plt2_img = Image.open(plt2_bytes)
    get_concat(plt1_img, plt2_img, direction).save(output_file)


# %%
###################################### Load & clean data
FIRST = True
for approach, data in mem_files.items():
    data_df = pd.read_csv(os.path.join(output_dir, data)).dropna()
    if ('model_size' in data_df.columns) and FIRST:
        mem_results = pd.DataFrame(
            columns=['approach', 'opt_level', 'model_size', 'num_vertices', 'features', 'mean_total_time'])
    elif FIRST:
        mem_results = pd.DataFrame(
            columns=['approach', 'opt_level', 'num_vertices', 'features', 'mean_total_time'])
    FIRST = False

    for uf in data_df.features.unique():
        feat_df = data_df[data_df.features == uf]
        for opt in data_df.opt_level.unique():
            opt_df = feat_df[feat_df.opt_level == opt]
            for v in data_df.num_vertices.unique():
                num_vert = opt_df[opt_df.num_vertices == v]
                if 'model_size' in num_vert.columns:
                    for mod in data_df.model_size.unique():
                        mod_size = num_vert[num_vert.model_size == mod]
                        mean = mod_size.mean()
                        output_dict = {'approach': approach, 'opt_level': opt, 'model_size': mod,
                                       'num_vertices': v, 'features': uf, 'mean_total_time': mean.total_time}
                        mem_results = mem_results.append(
                            output_dict, ignore_index=True)
                else:
                    mean = num_vert.mean()
                    output_dict = {'approach': approach, 'opt_level': opt,
                                   'num_vertices': v, 'features': uf, 'mean_total_time': mean.total_time}
                    mem_results = mem_results.append(
                        output_dict, ignore_index=True)

del approach, data, data_df, uf, feat_df, opt, opt_df, v, num_vert, mod, mod_size, mean, output_dict


# %%
###################################### Timing plot
if 'model_size' in mem_results.columns:
    gae_mem_plot = (ggplot(mem_results, aes(x='num_vertices', y='mean_total_time', group='opt_level', color='factor(opt_level)'))
                    + labs(color='Opt Level')
                    + geom_line()
                    + facet_wrap(('features', 'model_size', 'approach'), labeller="label_both")
                    ) + theme(figure_size=(20, 8), axis_text_x=element_text(rotation=45)) #+ scale_y_log10()
else:
    gae_mem_plot = (ggplot(mem_results, aes(x='num_vertices', y='mean_total_time', group='opt_level', color='factor(opt_level)'))
                    + labs(color='Opt Level')
                    + geom_line()
                    + facet_wrap(('features', 'approach'), labeller="label_both")
                    ) + theme(figure_size=(12, 4)) + scale_y_log10()
gae_mem_plot
#%%
gae_mem_plot.save(os.path.join(output_dir, mem_plot))


# %%
###################################### GAE accuracy plot
gae_acc_df = pd.read_csv(os.path.join(
    output_dir, gae_acc_file)).dropna().drop(['Unnamed: 0'], axis=1)

# %%
acc_roc = (ggplot(gae_acc_df) + stat_summary(aes('opt_level', 'roc_score', fill='factor(features)'), geom='col', position='dodge')
 + stat_summary(aes('opt_level', 'roc_score', fill='factor(features)'), fun_ymin=np.min,
                fun_ymax=np.max, geom='linerange', size=1, position=position_dodge(width=0.9))
 + facet_wrap(('padding'), labeller="label_both")
 ) + theme(figure_size=(12, 4))

acc_ap = (ggplot(gae_acc_df) + stat_summary(aes('opt_level', 'ap_score', fill='factor(features)'), geom='col', position='dodge')
 + stat_summary(aes('opt_level', 'ap_score', fill='factor(features)'), fun_ymin=np.min,
                fun_ymax=np.max, geom='linerange', size=1, position=position_dodge(width=0.9))
 + facet_wrap(('padding'), labeller="label_both")
 ) + theme(figure_size=(12, 4))

#%%
stack_save_plots(acc_roc, acc_ap, os.path.join(output_dir, gae_acc_plot), direction='v')
#%%
acc_roc
#%%
acc_ap


# %%
###################################### GCN accuracy plot
gcn_acc_df = pd.read_csv(os.path.join(
    output_dir, gcn_acc_file)).dropna().drop(['Unnamed: 0'], axis=1)

acc = (ggplot(gcn_acc_df) + stat_summary(aes('opt_level', 'acc_test', fill='factor(features)'), geom='col', position='dodge')
 + stat_summary(aes('opt_level', 'acc_test', fill='factor(features)'), fun_ymin=np.min,
                fun_ymax=np.max, geom='linerange', size=1, position=position_dodge(width=0.9))
 + facet_wrap(('padding'), labeller="label_both")
 ) + theme(figure_size=(12, 4))

#%%
acc.save(os.path.join(output_dir, gcn_acc_plot))

# %%


