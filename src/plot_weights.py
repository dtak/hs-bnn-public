import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import argparse
from sklearn.externals import joblib
from src.hs_bnn import HSBnn

sb.set_context("paper", rc={"lines.linewidth": 5, "lines.markersize":10, 'axes.labelsize': 8,
   'text.fontsize': 8,
   'legend.fontsize': 15,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'ylabel.fontsize':15,
   'xlabel.fontsize':15,
   'text.usetex': False,
    'axes.titlesize' : 25,
    'axes.labelsize' : 25,  })
sb.set_style("darkgrid")

def plot_singlelayer_weights(mlp, posterior_mode=False):
    plt.figure()
    axx = plt.gca()
    if mlp.inference_engine.classification:
        w_vect, sigma_vect, tau_mu_vect, tau_sigma_vect, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer = \
            mlp.inference_engine.unpack_params(mlp.optimal_elbo_params)
    else:
        w_vect, sigma_vect, tau_mu_vect, tau_sigma_vect, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer, _, _ = \
            mlp.inference_engine.unpack_params(mlp.optimal_elbo_params)
    for layer_id, (mu, var, tau_mu, tau_sigma) in enumerate(zip(mlp.inference_engine.unpack_layer_weights(w_vect),
                                                                mlp.inference_engine.unpack_layer_weights(
                                                                    sigma_vect),
                                                                mlp.inference_engine.unpack_layer_weight_priors(
                                                                    tau_mu_vect),
                                                                mlp.inference_engine.unpack_layer_weight_priors(
                                                                    tau_sigma_vect))):
        scale_mu = 0.5 * (tau_mu + tau_mu_global[layer_id])
        scale_v = 0.25 * (tau_sigma ** 2 + tau_sigma_global[layer_id] ** 2)
        w, b = mu
        if not posterior_mode:
            wstack = np.vstack([w, b]) * np.exp(scale_mu + np.sqrt(scale_v) / 2)
        else:
            wstack = np.vstack([w, b]) * np.exp(scale_mu - np.sqrt(scale_v))
        idx = np.argsort(np.linalg.norm(wstack, axis=0))
        if idx.shape[0] > 20:
            sb.boxplot(data=wstack[:, idx[-20:]], orient="h", ax=axx)
        else:
            sb.boxplot(data=wstack[:, idx], orient="h", ax=axx)
        plt.show(block=True)
        return wstack

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("r_path")
    args = parser.parse_args()
    r_path =args.r_path
    model = joblib.load(r_path)
    plot_singlelayer_weights(model)
