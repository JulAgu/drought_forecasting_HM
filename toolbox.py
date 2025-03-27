import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, f1_score, roc_auc_score, root_mean_squared_error
from tqdm import tqdm
import models


def charge_model(kind_of_model, model_path, hyperparams, dim_info):
    device=torch.device("cpu")
    print(f"Using device: {device}")
    print(torch.cuda.get_device_name(device=None))
    if kind_of_model == "HM":
        model = models.HybridModel(
            hyperparams["output_weeks"],
            dim_info["n_tf"],
            hyperparams["hidden_dim"],
            hyperparams["n_layers"],
            hyperparams["ffnn_layers"],
            hyperparams["dropout"],
            dim_info["static_dim"],
            dim_info["list_cat"],
            hyperparams["embed_dim"],
            hyperparams["embed_dropout"],
            hyperparams["ablation_TS"],
            hyperparams["ablation_tabular"],
            hyperparams["ablation_attention"],
        )
    elif kind_of_model == "LSTM":
        model = models.DroughtNetLSTM(
        hyperparams["output_weeks"],
        dim_info["n_tf"],
        hyperparams["hidden_dim"],
        hyperparams["n_layers"],
        hyperparams["ffnn_layers"],
        hyperparams["dropout"],
        dim_info["static_dim_lstm"],
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    return model, device


def predict(kind_of_model, model, data):
    if kind_of_model == "HM":
        val_h, x, static, cat = data
        out, _ = model(torch.tensor(x), val_h, static, cat)
    elif kind_of_model == "LSTM":
        val_h, x, static = data
        out, _ = model(torch.tensor(x), val_h, static)
    return out


def evaluation_loop(kind_of_model, model, device, loader, hyperparams):
    dict_map = {
    "y_pred": [],
    "y_pred_rounded": [],
    "y_true": [],
    "week": [],
    }
    i = 0
    if kind_of_model == "HM":
        for x, static, catego, y in tqdm(
            loader, # ou test_loader
            desc="predictions...",):
            val_h = tuple([each.data.to(device) for each in model.init_hidden(len(x), device)])
            x, static, catego, y = x.to(device), static.to(device), catego.to(device), y.to(device)
            with torch.no_grad():
                pred = predict(kind_of_model, model, (val_h, x, static, catego)).clone().detach()
            for w in range(hyperparams["output_weeks"]):
                dict_map["y_pred"] += [float(p[w]) for p in pred]
                dict_map["y_pred_rounded"] += [int(p.round()[w]) for p in pred]
                dict_map["y_true"] += [float(item[w]) for item in y]
                dict_map["week"] += [w] * len(x)
            i += len(x)
    elif kind_of_model == "LSTM":
        for x, static, y in tqdm(
        loader, # ou test_loader
        desc="predictions...",):   
            val_h = tuple([each.data.to(device) for each in model.init_hidden(len(x), device)])
            x, static, y = x.to(device), static.to(device), y.to(device)
            with torch.no_grad():
                pred = predict(kind_of_model, model, (val_h, x, static)).clone().detach()
            for w in range(hyperparams["output_weeks"]):
                dict_map["y_pred"] += [float(p[w]) for p in pred]
                dict_map["y_pred_rounded"] += [int(p.round()[w]) for p in pred]
                dict_map["y_true"] += [float(item[w]) for item in y]
                dict_map["week"] += [w] * len(x)
            i += len(x)
    df = pd.DataFrame(dict_map)
    return df
    
def week_evaluation_loop(df, print_results=False):
    # Create the list with the weekly metrics
    mae_list = []
    f1_list = []
    for w in range(6):
        wdf = df[df['week']==w]
        mae = mean_absolute_error(wdf['y_true'], wdf['y_pred']).round(3)
        f1 = f1_score(wdf['y_true'].round(),wdf['y_pred'].round(), average='macro').round(3)
        mae_list.append(mae)
        f1_list.append(f1)
        if print_results:
            print(f"Week {w+1}", f"MAE {mae}", f"F1 {f1}")
    print("----------------------------------")
    return mae_list, f1_list

def week_evaluation(df_val, df_test, master_dict):
    val_w_mae, val_w_f1 = week_evaluation_loop(df_val)
    test_w_mae, test_w_f1 = week_evaluation_loop(df_test, print_results=True)
    master_dict["w_val_mae"] = val_w_mae
    master_dict["w_val_f1"] = val_w_f1
    master_dict["w_test_mae"] = test_w_mae
    master_dict["w_test_f1"] = test_w_f1
    return master_dict

def overall_evaluation_loop(df):
    y_true_roc = df['y_true'].round()
    y_pred_roc = df['y_pred'].round()
    y_pred_for_sklearn = np.array([[0, 0, 0, 0, 0, 0] for i in y_pred_roc])
    for i in range(len(y_pred_roc)):
        y_pred_for_sklearn[i, int(y_pred_roc[i])] = 1

    y_true_for_sklearn = np.array([[0, 0, 0, 0, 0, 0] for i in y_true_roc])
    for i in range(len(y_true_roc)):
        y_true_for_sklearn[i, int(y_true_roc[i])] = 1
    mae = mean_absolute_error(df['y_true'], df['y_pred']).round(3)
    rmse = root_mean_squared_error(df['y_true'], df['y_pred']).round(3)
    f1 = f1_score(y_true_roc, y_pred_roc, average='macro').round(3)
    roc = roc_auc_score(y_true_for_sklearn, y_pred_for_sklearn, average='macro').round(3)
    return mae, rmse, f1, roc


def overall_evaluation(df_val, df_test, master_dict):
    val_mae, val_rmse, val_f1, val_roc = overall_evaluation_loop(df_val)
    test_mae, test_rmse, test_f1, test_roc = overall_evaluation_loop(df_test)
    master_dict["val_mae"] = val_mae
    master_dict["val_rmse"] = val_rmse
    master_dict["val_f1"] = val_f1
    master_dict["val_roc"] = val_roc
    master_dict["test_mae"] = test_mae
    master_dict["test_rmse"] = test_rmse
    master_dict["test_f1"] = test_f1
    master_dict["test_roc"] = test_roc
    return master_dict


def train(kind_of_model, hyperparams, dim_info,
          master_dict, tensor_path, model_path,
          train_loader, valid_loader):
    # Define the device
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("using GPU")
    else:
        device = torch.device("cpu")
        print("using CPU")
    torch.manual_seed(42)
    np.random.seed(42)
    if kind_of_model == "HM":
        model = models.HybridModel(
            hyperparams["output_weeks"],
            dim_info["n_tf"],
            hyperparams["hidden_dim"],
            hyperparams["n_layers"],
            hyperparams["ffnn_layers"],
            hyperparams["dropout"],
            dim_info["static_dim"],
            dim_info["list_cat"],
            hyperparams["embed_dim"],
            hyperparams["embed_dropout"],
            hyperparams["ablation_TS"],
            hyperparams["ablation_tabular"],
            hyperparams["ablation_attention"],
        )
    elif kind_of_model == "LSTM":
        model = models.DroughtNetLSTM(
        hyperparams["output_weeks"],
        dim_info["n_tf"],
        hyperparams["hidden_dim"],
        hyperparams["n_layers"],
        hyperparams["ffnn_layers"],
        hyperparams["dropout"],
        dim_info["static_dim_lstm"],
        )
    model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["lr"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=hyperparams["lr"],
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=hyperparams["epochs"])
    counter = 0
    writer = SummaryWriter(tensor_path)
    master_dict["tensorboard_path"] = tensor_path
    for i in range(hyperparams["epochs"]):
        h = model.init_hidden(hyperparams["batch_size"], device)
        for k, loader_content in tqdm(
        enumerate(train_loader),
        desc=f'epoch {i+1}/{hyperparams["epochs"]}',
        total=len(train_loader),
        ):
            counter +=1
            model.train()
            if kind_of_model == "HM":
                inputs, static, catego, labels = loader_content
                inputs, static, catego, labels = inputs.to(device), static.to(device), catego.to(device), labels.to(device)
            elif kind_of_model == "LSTM":
                inputs, static, labels = loader_content
                inputs, static, labels = inputs.to(device), static.to(device), labels.to(device)
            if len(inputs) < hyperparams["batch_size"]:
                h = model.init_hidden(len(inputs), device)
            h = tuple([e.data for e in h])
            model.zero_grad()
            if kind_of_model == "HM":
                output, h = model(inputs, h, static, catego)
            elif kind_of_model == "LSTM":
                output, h = model(inputs, h, static)
            loss = loss_function(output, labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hyperparams["clip"])
            optimizer.step()
            scheduler.step()
            # Log the validation loss
            with torch.no_grad():
                if k == len(train_loader) - 1 or k == (len(train_loader) - 1) // 2:
                    val_h = model.init_hidden(hyperparams["batch_size"], device)
                    val_losses = []
                    model.eval()
                    labels = []
                    preds = []
                    raw_labels = []
                    raw_preds = []
                    for load_cont in valid_loader:
                        if kind_of_model == "HM":
                            inp, stat, cat, lab = load_cont
                            inp, stat, cat, lab = inp.to(device), stat.to(device), cat.to(device), lab.to(device)
                        elif kind_of_model == "LSTM":
                            inp, stat, lab = load_cont
                            inp, stat, lab = inp.to(device), stat.to(device), lab.to(device)
                        if len(inp) < hyperparams["batch_size"]:
                            val_h = model.init_hidden(len(inp), device)
                        val_h = tuple([each.data for each in val_h])
                        if kind_of_model == "HM":
                            out, val_h = model(inp, val_h, stat, cat)
                        elif kind_of_model == "LSTM":
                            out, val_h = model(inp, val_h, stat)
                        val_loss = loss_function(out, lab.float())
                        val_losses.append(val_loss.item())
                        for labs in lab:
                            labels.append([int(l.round()) for l in labs])
                            raw_labels.append([float(l) for l in labs])
                        for pred in out:
                            preds.append([int(p.round()) for p in pred])
                            raw_preds.append([float(p) for p in pred])
                    raw_labels = np.array(raw_labels)
                    raw_preds = np.array(raw_preds)
                    labels = np.array(labels)
                    preds = np.clip(np.array(preds), 0, 5) # clip predictions to avoid errors when testing poor models.
                    log_dict = {
                        "loss": loss.item(),
                        "val_loss": np.mean(val_losses),
                        "val_macro_f1": f1_score(labels.flatten(), preds.flatten(), average="macro"),
                        "val_micro_f1": f1_score(labels.flatten(), preds.flatten(), average="micro"),
                        "val_mae": mean_absolute_error(raw_labels, raw_preds),
                        "epoch": counter/len(train_loader),
                        "step": counter,
                        "lr": optimizer.param_groups[0]["lr"],
                    }

                    writer.add_scalars("Loss_MSE", {"train": log_dict["loss"], "val": log_dict["val_loss"]},
                                       counter)
                    writer.add_scalars("F1", {"macro": log_dict["val_macro_f1"], "micro": log_dict["val_micro_f1"]},
                                       counter)
                    writer.add_scalar("MAE", log_dict["val_mae"],
                                      counter)
                    writer.add_scalar("Learning-rate", log_dict["lr"],
                                      counter)

                    torch.save(model.state_dict(), model_path)

                    print(f'EPOCH {i+1}/{hyperparams["epochs"]}...')
                    print(f'Step: {log_dict["step"]}... Loss: {log_dict["loss"]}\n')
                    print(f'Val Loss: {log_dict["val_loss"]}')
                    print("==================================\n")
    return master_dict
 
def evaluation(kind_of_model,
               hyperparams,
               dim_info,
               model_path,
               valid_loader,
               test_loader,
               master_dict):
    model, device = charge_model(kind_of_model, model_path, hyperparams, dim_info)
    model.eval()
    df_val = evaluation_loop(kind_of_model, model, device, valid_loader, hyperparams)
    df_test = evaluation_loop(kind_of_model, model, device, test_loader, hyperparams)
    week_evaluation(df_val, df_test, master_dict)
    overall_evaluation(df_val, df_test, master_dict)
    return master_dict