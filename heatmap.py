#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

#%%

fp_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
fn_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


baseline = 1 - np.array([[0.8335892 , 0.83143997, 0.82591343, 0.8038072 , 0.8038072 ,
        0.76481426],
       [0.83113295, 0.83573842, 0.8200798 , 0.82038689, 0.77126187,
        0.76082283],
       [0.82100093, 0.83727354, 0.83143997, 0.8286767 , 0.78569233,
        0.75161189],
       [0.8056494 , 0.81393921, 0.80534232, 0.82345718, 0.8200798 ,
        0.76020879],
       [0.65213388, 0.71139085, 0.79244703, 0.7740252 , 0.77463925,
        0.75683147],
       [0.46576604, 0.4009825 , 0.49554804, 0.61743939, 0.45716918,
        0.49831134]])
forward = 1 - np.array([[0.83850169, 0.83297515, 0.83451027, 0.83236104, 0.83727354,
        0.83143997],
       [0.83850169, 0.83236104, 0.83481729, 0.8305189 , 0.82407123,
        0.82161498],
       [0.83880872, 0.83113295, 0.83727354, 0.82652748, 0.82806265,
        0.82898372],
       [0.83573842, 0.83665949, 0.82929075, 0.82652748, 0.82898372,
        0.80933374],
       [0.82591343, 0.83481729, 0.82345718, 0.82407123, 0.79275405,
        0.76020879],
       [0.82345718, 0.83205402, 0.81823766, 0.81793064, 0.74762052,
        0.44273871]])
backward = 1 - np.array([[0.83573842, 0.83635247, 0.83082592, 0.83727354, 0.8286767 ,
        0.82468528],
       [0.83389622, 0.83880872, 0.82714152, 0.83573842, 0.82898372,
        0.82591343],
       [0.8354314 , 0.83297515, 0.82806265, 0.82345718, 0.82222903,
        0.81639546],
       [0.83143997, 0.831747  , 0.82315016, 0.82622045, 0.81915873,
        0.82315016],
       [0.83113295, 0.82929075, 0.82806265, 0.82775563, 0.7988947 ,
        0.80718452],
       [0.83021188, 0.8305189 , 0.83297515, 0.80810565, 0.74669939,
        0.59134173]])

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

