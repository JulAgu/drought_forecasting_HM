import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from utilities import (data_pretreatment_baseline,
                      create_simple_report)
from toolbox import train, evaluation

# Define the master_dictionary useful to create the report
master_dict = {"name": None,
               "hyperparameters": list(),
               }
# Define the hyperparameters dictionary
hyperparameters_dict = {}
# Define a last dictionary with the information about the dimensions of the model
dim_info = {}
# Define the hyperparameters
batch_size = 128
output_weeks = 6
hidden_dim = 512
n_layers = 2
ffnn_layers = 2
dropout = 0.1
lr = 7e-5
epochs = 7
clip = 5
# Add the hyperparameters to the dictionarys
hp_name_list = ["batch_size", "output_weeks",
                "hidden_dim", "n_layers",
                "ffnn_layers", "dropout",
                "lr", "epochs", "clip"]
for idx, hp in enumerate([batch_size, output_weeks,
                          hidden_dim, n_layers,
                          ffnn_layers, dropout,
                          lr, epochs, clip]):
    # Add the hyperparameters to the master dictionary
    master_dict["hyperparameters"].append(list())
    master_dict["hyperparameters"][idx].append(hp_name_list[idx])
    master_dict["hyperparameters"][idx].append(hp)
    # Add the hyperparameters to the hyperparameters dictionary
    hyperparameters_dict[hp_name_list[idx]] = hp

# Load the data
data = data_pretreatment_baseline()
X_tabular_train = data["X_tabular_train"]
X_time_train = data["X_time_train"]
y_target_train = data["y_target_train"]
X_tabular_valid = data["X_tabular_validation"]
X_time_valid = data["X_time_validation"]
y_target_valid = data["y_target_validation"]
valid_fips = data["valid_fips"]
X_tabular_test = data["X_tabular_test"]
X_time_test = data["X_time_test"]
y_target_test = data["y_target_test"]
test_fips = data["test_fips"]

train_data = TensorDataset(
torch.tensor(X_time_train),
torch.tensor(X_tabular_train),
torch.tensor(y_target_train[:, :output_weeks]),
)
train_loader = DataLoader(
    train_data, shuffle=True, batch_size=batch_size, drop_last=False
)
valid_data = TensorDataset(
    torch.tensor(X_time_valid),
    torch.tensor(X_tabular_valid),
    torch.tensor(y_target_valid[:, :output_weeks]),
)
valid_loader = DataLoader(
    valid_data, shuffle=False, batch_size=batch_size, drop_last=False
)

test_data = TensorDataset(
    torch.tensor(X_time_test),
    torch.tensor(X_tabular_test),
    torch.tensor(y_target_test[:, :output_weeks]),
)
test_loader = DataLoader(
    test_data, shuffle=False, batch_size=batch_size, drop_last=False
)
dim_info["static_dim_lstm"] = X_tabular_train.shape[-1]
dim_info["n_tf"] = X_time_train.shape[-1]

if __name__ == "__main__":
    EXPE_NAME = "LSTM_baseline_toolbox"
    master_dict["name"] = EXPE_NAME
    # Paths
    ROOT_RESULTS = f"results/{EXPE_NAME}/"
    ROOT_TENSORBOARD = f"runs/{EXPE_NAME}/"
    ROOT_MODELS_WEIGHTS = f"models/{EXPE_NAME}/"
    # Eliminate the previous results
    os.system(f"rm -rf {ROOT_RESULTS}")
    os.system(f"rm -rf {ROOT_TENSORBOARD}")
    os.system(f"rm -rf {ROOT_MODELS_WEIGHTS}")
    # Create the directories if they don't exist
    os.makedirs(ROOT_RESULTS)
    os.makedirs(ROOT_TENSORBOARD)
    os.makedirs(ROOT_MODELS_WEIGHTS)

    master_dict = train(kind_of_model="LSTM",
                        hyperparams=hyperparameters_dict,
                        dim_info=dim_info,
                        master_dict=master_dict,
                        tensor_path=ROOT_TENSORBOARD,
                        model_path=ROOT_MODELS_WEIGHTS + EXPE_NAME + ".pt",
                        train_loader=train_loader,
                        valid_loader=valid_loader,
                        )
    master_dict = evaluation(kind_of_model="LSTM",
                             hyperparams=hyperparameters_dict,
                             dim_info=dim_info,
                             model_path=ROOT_MODELS_WEIGHTS + EXPE_NAME + ".pt",
                             test_loader=test_loader,
                             valid_loader=valid_loader,
                             master_dict=master_dict,)
    create_simple_report(template_path="extra_tools/simple_template.md",
                         output_path= f"{ROOT_RESULTS}{EXPE_NAME}.md",
                         data=master_dict)

