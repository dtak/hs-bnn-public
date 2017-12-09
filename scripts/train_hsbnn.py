import time, os
import numpy as np
from sklearn.externals import joblib
from src.load_data import classification_data, regression_data
from src.factorized_approximation import FactorizedHierarchicalInvGamma as inference_engine
from src.hs_bnn import HSBnn, fit

###### Classification/Regression ##############
classification = True

#### ARCH Params ##########
num_hidden_layers = 1
num_nodes = 20

#### LEARNING Params ###########
batch_size = 512
learning_rate = 0.005
num_iterations = 5000
polyak = False  # polyak averaging

# sparsity parameter
lambda_b_global = 1

seed = 0
if classification:
    x_train, y_train, x_test, y_test, train_stats = classification_data(seed)
else:
    x_train, y_train, x_test, y_test, train_stats = regression_data(seed)
num_nodes_list = [num_nodes for i in np.arange(num_hidden_layers)]
if not classification:
    layer_sizes = [1] + num_nodes_list + [1]
else:
    layer_sizes = [2] + num_nodes_list + [2]
if batch_size > x_train.shape[0]:
    batch_size = x_train.shape[0]
num_epochs = int(np.ceil(num_iterations / (x_train.shape[0] / batch_size)))
print("Num Epochs {0} {1}".format(num_epochs, x_train.shape[0]))
mlp = HSBnn(layer_sizes, train_stats, x_train, y_train, x_test, y_test, inference_engine,
            classification=classification, batch_size=batch_size, lambda_b_global=lambda_b_global, polyak=polyak)
mlp = fit(mlp, n_epochs=num_epochs, l_rate=learning_rate)
save_date = time.strftime("%m_%d_%Y")
if classification:
    save_path = "./results/Classification/{0}".format(save_date)
else:
    save_path = "./results/Regression/{0}".format(save_date)
os.makedirs(save_path, exist_ok=True)
save_name = "{0}/hsbnn_{1}_{2}.pkl".format(save_path, num_nodes, lambda_b_global)
joblib.dump(mlp, save_name)