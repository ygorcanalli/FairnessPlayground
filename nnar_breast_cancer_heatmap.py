#%%
import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import sqlite3
import persistence
# sphinx_gallery_thumbnail_number = 2

#%%


def get_heatmap_data(df, fp_white, fn_white, black_fp_rates, black_fn_rates):
        black_df = df.loc[ (df.fp_white==fp_white) & 
                             (df.fn_white==fn_white),
                             ['fp_black', 'fn_black', 'test_acc']
                        ]
        black_acc = black_df.sort_values(by=['fn_black', 'fp_black']).test_acc.values

        heatmap_data = black_acc.reshape( (black_fp_rates.shape[0], black_fn_rates.shape[0]) )
        return heatmap_data
#%%
def plot_heatmap_ax(ax, data, fp_white, fn_white, black_fp_rates, black_fn_rates, title, color_map=plt.cm.Reds, vmin=None, vmax=None):
        im = ax.imshow(data,cmap=color_map)
        if vmin is not None and vmax is not None:
                im.set_clim(vmin, vmax)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(black_fp_rates)))
        ax.set_yticks(np.arange(len(black_fn_rates)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(["fp_black: %.1f" % fp for fp in black_fp_rates], fontsize=3, rotation=45)
        ax.set_yticklabels(["fn_black: %.1f" % fn for fn in black_fn_rates], fontsize=3)

        ax.set_xlabel("fp_white: %.1f" % fp_white, fontsize=6)
        ax.set_ylabel("fn_white: %.1f" % fn_white, fontsize=6)

        #Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), ha="right",
                rotation_mode="anchor")


        # Loop over data dimensions and create text annotations.
        
        for i in range(len(black_fn_rates)):
                for j in range(len(black_fp_rates)):
                        text = ax.text(j, i, "%.3f" % data[i, j] ,
                                ha="center", va="center", color="black", fontsize=1)

        #ax.set_title(title)


#%%

def plot_heatmap(data, title, difference_to=None, color_map=plt.cm.Reds, vmin=None, vmax=None):
        white_fp_rates = data.fp_white.sort_values().unique()
        white_fn_rates = data.fn_white.sort_values().unique()
        black_fp_rates = data.fp_black.sort_values().unique()
        black_fn_rates = data.fn_black.sort_values().unique()

               #fig.tight_layout()
        #fig = plt.figure(figsize=(180,180)) 
        plt.figure(figsize = (white_fn_rates.shape[0], white_fp_rates.shape[0]))
        fig, axs = plt.subplots(white_fn_rates.shape[0], white_fp_rates.shape[0], sharex='col', 
                                sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
        for i, ax_row in enumerate(axs):
                for j, ax in enumerate(ax_row):
                        heatmap_data = 1 - get_heatmap_data(data, white_fp_rates[j], white_fn_rates[i],
                                                        black_fp_rates, black_fn_rates)
                        if difference_to is not None:
                                diff_heatmap_data = 1 - get_heatmap_data(difference_to, white_fp_rates[j],
                                                        white_fn_rates[i], black_fp_rates, black_fn_rates)
                                difference = (diff_heatmap_data - heatmap_data)/diff_heatmap_data
                                plot_heatmap_ax(ax, difference, white_fp_rates[j], white_fn_rates[i], 
                                                        black_fp_rates, black_fn_rates, title, 
                                                        color_map=color_map, vmin=vmin, vmax=vmax)                              
                        else:
                                plot_heatmap_ax(ax, heatmap_data, white_fp_rates[j], white_fn_rates[i], 
                                                black_fp_rates, black_fn_rates, title,
                                                color_map=color_map, vmin=vmin, vmax=vmax)

        fig.subplots_adjust(wspace=0, hspace=0)
        fig.suptitle(title, fontsize=16)
        for ax in axs.flat:
                ax.label_outer()
                ax.set_aspect('equal')
                
        #plt.show()
                
        fig.savefig(title+".pdf", bbox_inches='tight')

#%%
# Read sqlite query results into a pandas DataFrame
directory = "results/0003"
db_path = os.path.join(directory, "baseline.db")
conn = persistence.create_connection(db_path)
with conn:
        baseline = pd.read_sql_query("SELECT fp_male as fp_white, fn_male as fn_white, fp_female as fp_black, fn_female as fn_black, test_acc from result", conn)

db_path = os.path.join(directory, "two_step_forward.db")
conn = persistence.create_connection(db_path)
with conn:
        two_step_forward = pd.read_sql_query("SELECT fp_male as fp_white, fn_male as fn_white, fp_female as fp_black, fn_female as fn_black, test_acc from result", conn)

db_path = os.path.join(directory, "alternating_forward.db")
conn = persistence.create_connection(db_path)
with conn:
        alternating_forward = pd.read_sql_query("SELECT fp_male as fp_white, fn_male as fn_white, fp_female as fp_black, fn_female as fn_black, test_acc from result", conn)

#%%

white_fp_rates = baseline.fp_white.sort_values().unique()
white_fn_rates = baseline.fn_white.sort_values().unique()
black_fp_rates = baseline.fp_black.sort_values().unique()
black_fn_rates = baseline.fn_black.sort_values().unique()

plot_heatmap(baseline, "Baseline error rate",
                vmin=0,vmax=0.6)
plot_heatmap(alternating_forward, "Alternating forward error rate",
                vmin=0,vmax=0.6)
plot_heatmap(alternating_forward, "Alternating forward improvement", 
                difference_to=baseline, color_map=plt.cm.RdBu,
                vmin=-0.72,vmax=0.72)
plot_heatmap(two_step_forward, "Two step forward error rate",
                vmin=0,vmax=0.6)
plot_heatmap(two_step_forward, "Two step forward improvement", 
                difference_to=baseline, color_map=plt.cm.RdBu,
                vmin=-0.72,vmax=0.72)

#%%
