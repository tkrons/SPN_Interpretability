# %%

from data.synthetic_data import generate_icecream_data, generate_gender_age_data
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from data import real_data
from interpretability import visualizations as vz
from simple_spn import spn_handler
from simple_spn import functions as fn
from interpretability.visualizations import *
from util import io
from pprint import pprint

from sklearn.datasets import make_classification, make_moons, make_blobs
from sklearn.manifold import TSNE
# %%
# data, target = make_classification(n_samples=20, n_features=2,)
# data, target = make_blobs(n_samples=40, n_features=2, random_state=6, cluster_std=3)
def plot_icecream(data):
    target = data[:, -1]
    if data.shape[1] <= 3:
        x, y = data[:, 0], data[:, 1]
        y = np.round(y - np.min(y))
        x = np.round((x - np.min(x) + 1) * 1.3)
        colors = {0: 'blue', 1: 'red', 2: 'green'}  # Mann, Frau, Kind
        plt.scatter(x, y, color=[colors[t] for t in target], )
        plt.yticks([0, 5, 10, 15, 20])
        plt.show()
    else:
        embedded = TSNE(random_state=1).fit_transform(data[:, :-1])
        colors = {0: 'blue', 1: 'red', 2: 'green'}  # Mann, Frau, Kind
        plt.scatter(embedded[:, 0], embedded[:, 1], color=[colors[t] for t in target],)
        plt.show()
num_vars = 6
data, value_dict, data_types = generate_icecream_data(200, num_vars - 1) # target doesnt count as feature...
dataset_name = 'icecream'
plot_icecream(data)

#plot corr matrix
df = pd.DataFrame(data, columns=[value_dict[i][1] for i in range(num_vars)])
print(df.corr())

# parameters for the construction
rdc_threshold = 0.1
min_instances_slice = 0.1
if not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice):
    print("Creating SPN ...")

    # get data
    # df, value_dict, parametric_types = real_data.get_titanic()

    spn, value_dict, _ = spn_handler.create_parametric_spns(data, data_types, dataset_name,
                                       [rdc_threshold], [min_instances_slice], value_dict, save=False)
# # Load SPN
# spn, value_dict, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)
# Print some statistics
fn.print_statistics(spn)
visualize_expected_sub_populations(spn, value_dict, 10)
visualize_sub_populations(spn, value_dict, 10)
subpops = fn.get_sub_populations(spn, )

print(subpops)
print('============')
pprint(subpops)
fn.plot_spn(spn, "icecream_spn.pdf", value_dict)
