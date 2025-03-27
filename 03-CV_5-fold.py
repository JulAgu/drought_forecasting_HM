import os
import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.model_selection import KFold
from utilities import (data_pretreatment_baseline,
                       data_pretreatment_hm,
                       create_simple_report)
from toolbox import train, evaluation

# Define the master_dictionary useful to create the report
master_dict_lstm = {"name": None,
                    "hyperparameters": list(),
                   }
master_dict = {"name": None,
               "hyperparameters": list(),
              }
# Define the hyperparameters dictionary
hyperparameters_dict_lstm = {}
hyperparameters_dict = {}
# Define a last dictionary with the information about the dimensions of the model
dim_info = {}
# Define the hyperparameters
batch_size = 128
output_weeks = 6
hidden_dim = 490
n_layers = 2
ffnn_layers = 2
dropout = 0.1
lr = 7e-5
epochs = 9
clip = 5
embed_dim = [3, 3, 3, 3, 3, 3, 3]
embed_dropout = 0.4
# Define the LSTM specific hyperparameters
hidden_dim_lstm = 512
epochs_lstm = 7
# Add the hyperparameters to the dictionaries
#MH
hp_name_list = ["batch_size", "output_weeks",
                "hidden_dim", "n_layers",
                "ffnn_layers", "dropout",
                "lr", "epochs", "clip",
                "embed_dim", "embed_dropout"]
for idx, hp in enumerate([batch_size, output_weeks,
                          hidden_dim, n_layers,
                          ffnn_layers, dropout,
                          lr, epochs, clip,
                          embed_dim, embed_dropout]):
    # Add the hyperparameters to the master dictionary
    master_dict["hyperparameters"].append(list())
    master_dict["hyperparameters"][idx].append(hp_name_list[idx])
    master_dict["hyperparameters"][idx].append(hp)
    # Add the hyperparameters to the hyperparameters dictionary
    hyperparameters_dict[hp_name_list[idx]] = hp

hyperparameters_dict["ablation_TS"] = False
hyperparameters_dict["ablation_tabular"] = False
hyperparameters_dict["ablation_attention"] = False

#LSTM
hp_lstm_name_list = ["batch_size", "output_weeks",
                     "hidden_dim", "n_layers",
                     "ffnn_layers", "dropout",
                     "lr", "epochs", "clip"]
for idx, hp in enumerate([batch_size, output_weeks,
                          hidden_dim_lstm, n_layers,
                          ffnn_layers, dropout,
                          lr, epochs_lstm, clip]):
    # Add the hyperparameters to the master dictionary
    master_dict_lstm["hyperparameters"].append(list())
    master_dict_lstm["hyperparameters"][idx].append(hp_name_list[idx])
    master_dict_lstm["hyperparameters"][idx].append(hp)
    # Add the hyperparameters to the hyperparameters dictionary
    hyperparameters_dict_lstm[hp_lstm_name_list[idx]] = hp

#Load the LSTM data
# Load the data
print("Loading the data assuming all the variables as numerical...")
data_lstm = data_pretreatment_baseline()
X_tabular_train = data_lstm["X_tabular_train"]
X_time_train = data_lstm["X_time_train"]
y_target_train = data_lstm["y_target_train"]
X_tabular_valid = data_lstm["X_tabular_validation"]
X_time_valid = data_lstm["X_time_validation"]
y_target_valid = data_lstm["y_target_validation"]
valid_fips = data_lstm["valid_fips"]
X_tabular_test = data_lstm["X_tabular_test"]
X_time_test = data_lstm["X_time_test"]
y_target_test = data_lstm["y_target_test"]
test_fips = data_lstm["test_fips"]

train_data_lstm = TensorDataset(
torch.tensor(X_time_train),
torch.tensor(X_tabular_train),
torch.tensor(y_target_train[:, :output_weeks]),
)
train_loader_lstm = DataLoader(
    train_data_lstm, shuffle=True, batch_size=batch_size, drop_last=False
)
valid_data_lstm = TensorDataset(
    torch.tensor(X_time_valid),
    torch.tensor(X_tabular_valid),
    torch.tensor(y_target_valid[:, :output_weeks]),
)
valid_loader_lstm = DataLoader(
    valid_data_lstm, shuffle=False, batch_size=batch_size, drop_last=False
)

test_data_lstm = TensorDataset(
    torch.tensor(X_time_test),
    torch.tensor(X_tabular_test),
    torch.tensor(y_target_test[:, :output_weeks]),
)
test_loader_lstm = DataLoader(
    test_data_lstm, shuffle=False, batch_size=batch_size, drop_last=False
)

# Load the data
print("Loading the data assuming saving the ordinal variables as categorical...")
data = data_pretreatment_hm()
X_tabular_train = data["X_tabular_train"]
X_tabular_cat_train = data["X_tabular_cat_train"]
X_time_train = data["X_time_train"]
y_target_train = data["y_target_train"]
X_tabular_valid = data["X_tabular_validation"]
X_tabular_cat_valid = data["X_tabular_cat_validation"]
X_time_valid = data["X_time_validation"]
y_target_valid = data["y_target_validation"]
valid_fips = data["valid_fips"]
X_tabular_test = data["X_tabular_test"]
X_tabular_cat_test = data["X_tabular_cat_test"]
X_time_test = data["X_time_test"]
y_target_test = data["y_target_test"]
test_fips = data["test_fips"]

train_data = TensorDataset(
    torch.tensor(X_time_train),
    torch.tensor(X_tabular_train),
    torch.tensor(X_tabular_cat_train),
    torch.tensor(y_target_train[:, :output_weeks]),
)
valid_data = TensorDataset(
    torch.tensor(X_time_valid),
    torch.tensor(X_tabular_valid),
    torch.tensor(X_tabular_cat_valid),
    torch.tensor(y_target_valid[:, :output_weeks]),
)

train_loader = DataLoader(
    train_data, batch_size=batch_size, drop_last=False
)

valid_loader = DataLoader(
    valid_data, shuffle=False, batch_size=batch_size, drop_last=False
)

test_data = TensorDataset(
    torch.tensor(X_time_test),
    torch.tensor(X_tabular_test),
    torch.tensor(X_tabular_cat_test),
    torch.tensor(y_target_test[:, :output_weeks]),
)

test_loader = DataLoader(
    test_data, shuffle=False, batch_size=batch_size, drop_last=False
)

# Concat the datasets
dataset = ConcatDataset([train_data, valid_data, test_data])
dataset_lstm = ConcatDataset([train_data_lstm, valid_data_lstm, test_data_lstm])

print (len(dataset), len(dataset_lstm))

K_folds = 5
kf = KFold(n_splits=K_folds, shuffle=True, random_state=42)

print("-------Starting the 5-fold CV--------------")

def reset_weights(m):
  '''
    Resetting model weights to avoid weight leakage.
    from: https://github.com/SPTAU/PyTorch-Learning/blob/main/KFold_learning.py
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


def wrapper_cv():
    """
    A wrapper to repeat the train-evaluation cycle in an ablative way.
    """
    global master_dict
    global master_dict_lstm
    results_lstm = []
    results_hm = []
    for fold, (train_index, test_index) in enumerate(kf.split(dataset)):        
        print(f"Fold {fold+1}")
        # expe_name = f"{kind_of_model}_{expe_name}_{fold}"
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)

        # in_contex_dataloaders
        train_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=train_subsampler
        )

        test_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=test_subsampler
        )
        train_loader_lstm = DataLoader(
            dataset_lstm, batch_size=batch_size, sampler=train_subsampler
        )

        test_loader_lstm = DataLoader(
            dataset_lstm, batch_size=batch_size, sampler=test_subsampler
        )
        # To don't change the code and as the validation set is not used in the training (used only for printing the validation loss)
        # I use the test set as validation and test set... Is not the best, but is the fastest way to do without refactor toolbox.py
        X_tabular_train = []
        X_tabular_categories_train = []
        for X_time_train, X_tab_train, X_tabular_cat_train, _ in train_loader:
            X_tabular_train.append(X_tab_train)
            X_tabular_categories_train.append(X_tabular_cat_train)
        X_tabular_train = torch.cat(X_tabular_train, dim=0).numpy()
        X_tabular_cat_train = torch.cat(X_tabular_categories_train, dim=0).numpy()
        for _, X_tab_train, _ in train_loader_lstm:
            X_tabular_train_lstm = X_tab_train
            
        dim_info["static_dim_lstm"] = X_tabular_train_lstm.shape[-1]
        dim_info["static_dim"] = X_tabular_train.shape[-1]
        dim_info["n_tf"] = X_time_train.shape[-1]
        dim_info["list_cat"] = [len(np.unique(X_tabular_cat_train[:,i])) + 1 for i in range(X_tabular_cat_train.shape[1])]
        master_dict_lstm["name"] = f"LSTM_{fold}"
        master_dict_lstm = train("LSTM",
                        hyperparameters_dict_lstm,
                        dim_info,
                        master_dict_lstm,
                        ROOT_TENSORBOARD + f"LSTM_{fold}",
                        ROOT_MODELS_WEIGHTS + f"LSTM_{fold}" + ".pt",
                        train_loader_lstm,
                        test_loader_lstm,
                        )
        master_dict_lstm = evaluation("LSTM",
                                hyperparameters_dict_lstm,
                                dim_info,
                                ROOT_MODELS_WEIGHTS + f"LSTM_{fold}" + ".pt",
                                test_loader_lstm,
                                test_loader_lstm,
                                master_dict_lstm,)
        create_simple_report(template_path="extra_tools/simple_template.md",
                            output_path= f"{ROOT_RESULTS}LSTM_{fold}.md",
                            data=master_dict_lstm)
        results_lstm.append((master_dict_lstm["test_mae"], master_dict_lstm["test_rmse"], master_dict_lstm["test_f1"]))
        master_dict["name"] = f"HM_{fold}"
        master_dict = train("HM",
                        hyperparameters_dict,
                        dim_info,
                        master_dict,
                        ROOT_TENSORBOARD + f"HM_{fold}",
                        ROOT_MODELS_WEIGHTS + f"HM_{fold}" + ".pt",
                        train_loader,
                        test_loader,
                        )
        master_dict = evaluation("HM",
                                hyperparameters_dict,
                                dim_info,
                                ROOT_MODELS_WEIGHTS + f"HM_{fold}" + ".pt",
                                test_loader,
                                test_loader,
                                master_dict,)
        create_simple_report(template_path="extra_tools/simple_template.md",
                            output_path= f"{ROOT_RESULTS}HM_{fold}.md",
                            data=master_dict)
        results_hm.append((master_dict["test_mae"], master_dict["test_rmse"], master_dict["test_f1"]))
    results_lstm = sorted(results_lstm, key=lambda x: x[-1])
    results_hm = sorted(results_hm, key=lambda x: x[-1])
    return results_lstm, results_hm

def perform_t_paired_test(results_lstm, results_hm):
    """
    Perform a t-paired test between the results of the LSTM and the HM.
    Reminder the list of tuples order are (mae, rmse, f1)
    """
    t_paired_results = {"mae_lstm": [x[0] for x in results_lstm],
                        "rmse_lstm": [x[1] for x in results_lstm],
                        "f1_lstm": [x[2] for x in results_lstm],
                        "mae_hm": [x[0] for x in results_hm],
                        "rmse_hm": [x[1] for x in results_hm],
                        "f1_hm": [x[2] for x in results_hm],
                        "t_statistic": [], "p_value": [],}
    # Perform the t-paired test
    for idx in range(3):
        t_statistic, p_value = stats.ttest_rel([x[idx] for x in results_lstm], [x[idx] for x in results_hm])
        t_paired_results["t_statistic"].append(t_statistic)
        t_paired_results["p_value"].append(p_value)
        create_simple_report(template_path="extra_tools/cv_template.md",
                            output_path= f"{ROOT_RESULTS}_summary_cv.md",
                            data=t_paired_results)

if __name__ == "__main__":
    # Paths
    ROOT_RESULTS = f"results/cv_5-fold/"
    ROOT_TENSORBOARD = f"runs/cv_5-fold/"
    ROOT_MODELS_WEIGHTS = f"models/cv_5-fold/"
    # Eliminate the previous results
    os.system(f"rm -rf {ROOT_RESULTS}")
    os.system(f"rm -rf {ROOT_TENSORBOARD}")
    os.system(f"rm -rf {ROOT_MODELS_WEIGHTS}")
    # Create the directories if they don't exist
    os.makedirs(ROOT_RESULTS)
    os.makedirs(ROOT_TENSORBOARD)
    os.makedirs(ROOT_MODELS_WEIGHTS)

    results_lstm, results_hm = wrapper_cv()
    perform_t_paired_test(results_lstm, results_hm)
    

