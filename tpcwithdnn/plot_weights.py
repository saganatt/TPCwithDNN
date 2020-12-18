from __future__ import print_function

import sys
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_conv_weights(conv_weights, layer, p_name, k_name):
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = conv_weights.min(), conv_weights.max()
    conv_weights = (conv_weights - f_min) / (f_max - f_min)
    out_channels = conv_weights.shape[-1]
    # n_conv_weights = outgoing channels
    n_conv_weights = out_channels
    for i in range(n_conv_weights):
        # get the filter
        f = conv_weights[:, :, :, :, i]
        # plot each channel separately
        # Range of in channels
        in_channels = f.shape[-1]
        # Range of depth of the kernel i.e. 3
        depth = f.shape[-2]
        fig, axs = plt.subplots(in_channels, depth, figsize=(4 * depth + 2, 2.5 * in_channels),
                                sharex=True, sharey=True, squeeze=False)
        for j in range(in_channels):
            for k in range(depth):
                sns.heatmap(ax=axs[j, k], data=f[:, :, k, j], annot=True, square=True, cmap="Blues")
                axs[j, k].set_xticks([])
                axs[j, k].set_yticks([])
                axs[j, k].set_title('input channel: {} depth: {}'.format(j, k))
        fig.suptitle('Convolution filters for out channel: {} layer: {}:{}:{}'
                     .format(i, layer, p_name, k_name))
        fig.tight_layout()
        fig.savefig('weights/weights_{}_{}_{}_out{}.png'
                    .format(layer, p_name, k_name, i),
                    bbox_inches='tight')
        plt.close()

def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.
    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path, 'r')
    try:
        if f.attrs.items():
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")

        print("  f.attrs.items(): ")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            print("  Terminate # len(f.items())==0: ")
            return

        print("  layer, g in f.items():")
        for layer, g in f.items():
            print("  {}".format(layer))
            print("    g.attrs.items(): Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("    Dataset: param.keys():")
                for k_name in param.keys():
                    weights = param.get(k_name)[:]
                    print("      {}/{}: {}".format(p_name, k_name, weights.shape))
                    print(weights)
                    #if (weights.ndim > 1 and 'conv3d_' in p_name and
                    #        weights.shape[-1] < 20 and weights.shape[-2] < 20
                    #        and weights.shape[-3] < 20):
                    #    plot_conv_weights(weights, layer, p_name, k_name)
    finally:
        f.close()

def compare_weights(file1, file2):
    f1 = h5py.File(file1, 'r')
    f2 = h5py.File(file2, 'r')
    try:
        for (layer1, g1), (layer2, g2) in zip(f1.items(), f2.items()):
            print("first: {} second: {}".format(layer1, layer2))
            for p_name1, p_name2 in zip(g1.keys(), g2.keys()):
                param1 = g1[p_name1]
                param2 = g2[p_name2]
                for k_name1, k_name2 in zip(param1.keys(), param2.keys()):
                    weights1 = param1.get(k_name1)[:]
                    weights2 = param2.get(k_name2)[:]
                    print("First: {}/{}: {}".format(p_name1, k_name1, weights1.shape))
                    print("Second: {}/{}: {}".format(p_name2, k_name2, weights2.shape))
                    diff_weights = np.abs(weights1 - weights2)
                    #print("Difference:\n{}".format(diff_weights))
                    print("Mean difference: {}".format(np.mean(diff_weights)))
                    #if (weights.ndim > 1 and 'conv3d_' in p_name and
                    #        weights.shape[-1] < 20 and weights.shape[-2] < 20
                    #        and weights.shape[-3] < 20):
                    #    plot_conv_weights(weights, layer, p_name, k_name)
    finally:
        f1.close()
        f2.close()

if len(sys.argv) > 2:
    compare_weights(sys.argv[1], sys.argv[2])
else:
    print_structure(sys.argv[1])
