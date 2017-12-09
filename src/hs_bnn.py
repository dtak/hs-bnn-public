from copy import copy

import autograd.numpy as np
from autograd import grad
from src.optimizers import adam
from src.utility_functions import make_batches


class HSBnn:
    def __init__(self, layer_sizes, train_stats, x, y, x_test, y_test, inference_engine, classification=False,
                 batch_size=128, lambda_b_global=1.0, warm_up=False, polyak=False):
        self.shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        self.layer_sizes = layer_sizes
        self.lambda_b_global = lambda_b_global
        self.N_weights = sum((m+1)*n for m, n in self.shapes)
        self.batches = make_batches(x.shape[0], batch_size)
        self.M = len(self.batches)  # Total number of batches
        self.elbo = list()
        self.val_ll = list()
        self.val_err = list()
        self.train_err = list()
        self.variational_params = None
        self.init_params = None
        self.polyak_params = None
        self.polyak = polyak
        self.variational_params_store = {}
        self.optimal_elbo_params = None
        self.warm_up = warm_up  # if True, anneal in KL
        self.X = x
        self.y = y
        self.X_test = x_test
        self.y_test = y_test
        self.inference_engine = inference_engine(lambda_a=0.5, lambda_b=1.0,
                                                 lambda_b_global=self.lambda_b_global, tau_a=0.5,
                                                 shapes=self.shapes, train_stats=train_stats,
                                                 classification=classification, n_data=self.X.shape[0],
                                                 n_weights=self.N_weights)

    def neg_elbo(self, params, epoch, x, y):
        if self.warm_up:
            nt = 200  # linear increments between 0 and 1 up to nt (1 after nt)
            temperature = epoch/nt
            if temperature > 1:
                temperature = 1
        else:
            temperature = 1
        log_lik, log_prior, ent_w, ent_tau, ent_lam = self.inference_engine.compute_elbo_contribs(params, x, y)
        log_variational = ent_w + ent_tau + ent_lam
        minibatch_rescaling = 1./self.M
        ELBO = temperature * minibatch_rescaling * (log_variational + log_prior) + log_lik
        return -1*ELBO

    def variational_objective(self, params, t):
        idx = self.batches[t % self.M]
        return self.neg_elbo(params, t/self.M, self.X[idx], self.y[idx])

    def compute_optimal_test_ll(self, num_samples=100):
        if not self.polyak:
            return self.inference_engine.compute_test_ll(self.variational_params, self.X_test,
                                                         self.y_test, num_samples=num_samples)
        else:
            return self.inference_engine.compute_test_ll(self.polyak_params, self.X_test,
                                                         self.y_test, num_samples=num_samples)


def fit(model, n_epochs=10, l_rate=0.01):
    def callback(params, t, g, decay=0.999):
        if model.polyak:
            # exponential moving average.
            model.polyak_params = decay * model.polyak_params + (1 - decay) * params
        score = -model.variational_objective(params, t)
        model.elbo.append(score)
        if (t % model.M) == 0:
            if model.polyak:
                val_ll, val_err = model.inference_engine.compute_test_ll(model.polyak_params, model.X_test, model.y_test)
            else:
                val_ll, val_err = model.inference_engine.compute_test_ll(params, model.X_test, model.y_test)
            train_err = model.inference_engine.compute_train_err(params, model.X, model.y)
            model.val_ll.append(val_ll)
            model.val_err.append(val_err)
            model.train_err.append(train_err)
            if ((t / model.M) % 10) == 0:
                if model.inference_engine.classification:
                    print("Epoch {} lower bound {} train_err {} test_err {} ".format(t/model.M, model.elbo[-1],
                                                                                     train_err,
                                                                                     model.val_err[-1]))
                else:
                    print("Epoch {} lower bound {} train_rmse {} test_rmse {} ".format(t / model.M, model.elbo[-1],
                                                                                     train_err, model.val_err[-1]))
            # randomly permute batch ordering every epoch
            model.batches = np.random.permutation(model.batches)
        if (t % 250) == 0:
            # store optimization progress.
            model.variational_params_store[t] = copy(params)
        if t > 2:
            if model.elbo[-1] > max(model.elbo[:-1]):
                model.optimal_elbo_params = copy(params)
        # update inverse gamma distributions
        model.inference_engine.fixed_point_updates(params)

    init_var_params = model.inference_engine.initialize_variational_params()
    model.init_params = copy(init_var_params)
    if model.polyak:
        model.polyak_params = copy(init_var_params)
    gradient = grad(model.variational_objective, 0)
    num_iters = n_epochs * model.M  # one iteration = one set of param updates
    model.variational_params = adam(gradient, init_var_params,
                                    step_size=l_rate, num_iters=num_iters, callback=callback,
                                    polyak=model.polyak)
    return model

