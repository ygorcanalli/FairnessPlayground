#%%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sqlite3
# sphinx_gallery_thumbnail_number = 2

#%%


def get_heatmap_data(df, fp_male, fn_male, female_fp_rates, female_fn_rates):
        female_df = df.loc[ (df.fp_male==fp_male) & 
                             (df.fn_male==fn_male),
                             ['fp_female', 'fn_female', 'test_acc']
                        ]
        female_acc = female_df.sort_values(by=['fp_female', 'fn_female']).test_acc.as_matrix()

        heatmap_data = female_acc.reshape( (female_fp_rates.shape[0], female_fn_rates.shape[0]) )
        return heatmap_data
#%%
def plot_heatmap_ax(ax, data, fp_male, fn_male, female_fp_rates, female_fn_rates, title, color_map=plt.cm.Reds, vmin=None, vmax=None):
        im = ax.imshow(data,cmap=color_map)
        if vmin is not None and vmax is not None:
                im.set_clim(vmin, vmax)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(female_fp_rates)))
        ax.set_yticks(np.arange(len(female_fn_rates)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(female_fp_rates, fontsize=3)
        ax.set_yticklabels(female_fn_rates, fontsize=3)

        ax.set_xlabel(fn_male)
        ax.set_ylabel(fp_male)

        # Rotate the tick labels and set their alignment.
        #plt.setp(ax.get_xticklabels(), ha="right",
        #        rotation_mode="anchor")


        # Loop over data dimensions and create text annotations.
        
        for i in range(len(female_fp_rates)):
                for j in range(len(female_fn_rates)):
                        text = ax.text(j, i, "%.3f" % data[i, j] ,
                                ha="center", va="center", color="black", fontsize=1)

        #ax.set_title(title)


#%%

def plot_heatmap(data, title, color_map=plt.cm.Reds, vmin=None, vmax=None):
        
        male_fp_rates = data.fp_male.sort_values().unique()
        male_fn_rates = data.fn_male.sort_values().unique()
        female_fp_rates = data.fp_female.sort_values().unique()
        female_fn_rates = data.fn_female.sort_values().unique()

        #fig = plt.figure(figsize=(180,180)) 
        plt.figure(figsize = (male_fn_rates.shape[0], male_fp_rates.shape[0]))
        fig, axs = plt.subplots(male_fn_rates.shape[0], male_fp_rates.shape[0], sharex='col', sharey='row',
                                gridspec_kw={'hspace': 0, 'wspace': 0})
        for i, ax_row in enumerate(axs):
                for j, ax in enumerate(ax_row):
                        heatmap_data = get_heatmap_data(data, male_fp_rates[i], male_fn_rates[j], female_fp_rates, female_fn_rates)
                        plot_heatmap_ax(ax, heatmap_data, male_fp_rates[i], male_fn_rates[j], female_fp_rates, female_fn_rates, title, color_map=color_map, vmin=vmin, vmax=vmax)

        #fig.tight_layout()
        for ax in axs.flat:
                ax.label_outer()
                ax.set_aspect('equal')

        fig.subplots_adjust(wspace=0, hspace=0)
                
        plt.show()
        fig.savefig(title+".pdf", bbox_inches='tight')

#%%
# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect("results/0007/result.db")
baseline = pd.read_sql_query("SELECT * from baseline", con)
alternating_forward = pd.read_sql_query("SELECT * from alternating_forward", con)
two_step_forward = pd.read_sql_query("SELECT * from two_step_forward", con)
con.close()

#%%

male_fp_rates = baseline.fp_male.sort_values().unique()
male_fn_rates = baseline.fn_male.sort_values().unique()
female_fp_rates = baseline.fp_female.sort_values().unique()
female_fn_rates = baseline.fn_female.sort_values().unique()

baseline_heatmap_data = get_heatmap_data(baseline,male_fp_rates[0], male_fn_rates[0], female_fp_rates, female_fn_rates)
alternating_forward_heatmap_data = get_heatmap_data(alternating_forward,male_fp_rates[0], male_fn_rates[0], female_fp_rates, female_fn_rates)
two_step_forward_heatmap_data = get_heatmap_data(two_step_forward,male_fp_rates[0], male_fn_rates[0], female_fp_rates, female_fn_rates)

alternating_forward_improvement = (alternating_forward_heatmap_data - baseline_heatmap_data)/baseline_heatmap_data
two_step_forward_improvement = (two_step_forward_heatmap_data - baseline_heatmap_data)/baseline_heatmap_data

plot_heatmap(baseline_heatmap_data, female_fp_rates, female_fn_rates, "Baseline error rate", vmin=0, vmax=0.6)
plot_heatmap(alternating_forward_heatmap_data, female_fp_rates, female_fn_rates, "Alternating forward error rate", vmin=0, vmax=0.6)
plot_heatmap(alternating_forward_improvement, female_fp_rates, female_fn_rates, "Alternating forward improvement", vmin=0, vmax=0.6)
plot_heatmap(two_step_forward_heatmap_data, female_fp_rates, female_fn_rates, "Two step forward error rate", vmin=0, vmax=0.6)
plot_heatmap(two_step_forward_improvement, female_fp_rates, female_fn_rates, "Two step forward improvement", vmin=0, vmax=0.6)

#%%
