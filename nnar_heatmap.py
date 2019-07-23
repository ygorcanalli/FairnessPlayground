#%%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

#%%

male_fp_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
male_fn_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
female_fp_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
female_fn_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

baseline = pd.read_csv("results/0006/baseline.csv", delimiter=",")
alternating_forward_half = pd.read_csv("results/0006/baseline.csv", delimiter=",")
#%%

def get_heatmap_data(df, fp_male, fn_male):
        data = np.zeros( (len(female_fp_rates), len(female_fn_rates)) )
        female_acc = df.loc[ (df.fp_male==fp_male) & 
                             (df.fn_male==fn_male),
                             ['fp_female', 'fn_female', 'acc']
                        ]
        return female_acc
#%%
def plot_heatmap(data, title, color_map=plt.cm.Reds, vmin=None, vmax=None):
        fig, ax = plt.subplots()
        im = ax.imshow(data,cmap=color_map)
        if vmin is not None and vmax is not None:
                im.set_clim(vmin, vmax)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(fp_rate)))
        ax.set_yticks(np.arange(len(fn_rate)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(fp_rate)
        ax.set_yticklabels(fn_rate)

        ax.set_xlabel("False positive rate")
        ax.set_ylabel("False negative rate")

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), ha="right",
                rotation_mode="anchor")


        # Loop over data dimensions and create text annotations.
        for i in range(len(fp_rate)):
                for j in range(len(fn_rate)):
                        text = ax.text(j, i, "%.3f" % data[i, j] ,
                                ha="center", va="center", color="black")

        ax.set_title(title)
        fig.tight_layout()
        plt.show()
        fig.savefig(title+".pdf", bbox_inches='tight')

#%%
plot_heatmap(baseline, "Baseline error rate", vmin=0, vmax=0.6)
plot_heatmap(forward, "Forward error rate", vmin=0, vmax=0.6)
forward_improvement = (baseline - forward)/baseline
plot_heatmap(forward_improvement, "Forward improvement", color_map=plt.cm.RdBu, vmin=-0.72, vmax=0.72)
plot_heatmap(backward, "Backward error rate", vmin=0, vmax=0.6)
backward_improvement = (baseline - backward)/baseline
plot_heatmap(backward_improvement, "Backward improvement", color_map=plt.cm.RdBu, vmin=-0.72, vmax=0.72)
#%%

