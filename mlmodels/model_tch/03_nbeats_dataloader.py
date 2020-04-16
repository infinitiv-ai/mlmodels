import json
import os

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import functional as F
from dataloader import DataLoader

####################################################################################################
from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri

VERBOSE = False
MODEL_URI = get_model_uri(__file__)


####################################################################################################
from mlmodels.model_tch.raw.nbeats.model import NBeatsNet

# Model
def Model(model_pars, data_pars, compute_pars):
    model_pars.update({"device": torch.device("cpu")})
    return NBeatsNet(**model_pars)


####################################################################################################
# Dataaset
def get_dataset(data_pars):
    loader = DataLoader(data_pars)
    loader.compute()
    data = loader.get_data()
    [print(x.shape) for x in data]
    return data


def data_generator(x_full, y_full, bs):
    def split(arr, size):
        arrays = []
        while len(arr) > size:
            slice_ = arr[:size]
            arrays.append(slice_)
            arr = arr[size:]
        arrays.append(arr)
        return arrays

    while True:
        for rr in split((x_full, y_full), bs):
            yield rr


######################################################################################################
# Model fit
def fit(model, data_pars=None, compute_pars=None, out_pars=None, **kw):
    device = torch.device("cpu")
    batch_size = compute_pars["batch_size"]  # greater than 4 for viz
    disable_plot = compute_pars["disable_plot"]

    ### Get Data
    x_train, X_test, y_train, y_test = get_dataset(data_pars)
    data_gen = data_generator(x_train, y_train, batch_size)

    ### Setup session
    optimiser = optim.Adam(model.parameters())

    ### fit model
    net, optimiser = fit_simple(
        model, optimiser, data_gen, plot_model, device, data_pars, out_pars
    )
    return net, optimiser


def fit_simple(
    net,
    optimiser,
    data_generator,
    on_save_callback,
    device,
    data_pars,
    out_pars,
    max_grad_steps=500,
):
    print("--- fiting ---")
    initial_grad_step = load_checkpoint(net, optimiser)

    for grad_step, (x, target) in enumerate(data_generator):
        grad_step += initial_grad_step
        optimiser.zero_grad()
        net.train()
        backcast, forecast = net(torch.tensor(x, dtype=torch.float).to(device))
        loss = F.mse_loss(forecast, torch.tensor(target, dtype=torch.float).to(device))
        loss.backward()
        optimiser.step()

        print(f"grad_step = {str(grad_step).zfill(6)}, loss = {loss.item():.6f}")
        if grad_step % 100 == 0 or (grad_step < 100 and grad_step % 100 == 0):
            with torch.no_grad():
                save_checkpoint(net, optimiser, grad_step)
                if on_save_callback is not None:
                    on_save_callback(net, x, target, grad_step, data_pars)

        if grad_step > max_grad_steps:
            print("Finished.")
            break
    return net, optimiser


def predict(model, sess, data_pars=None, compute_pars=None, out_pars=None, **kw):
    data_pars["train_split_ratio"] = 1

    x_train, x_test, y_train, y_test = get_dataset(**data_pars)

    test_losses = []
    model.eval()
    _, f = model(torch.tensor(x_test, dtype=torch.float))
    test_losses.append(F.mse_loss(f, torch.tensor(y_test, dtype=torch.float)).item())
    p = f.detach().numpy()
    return p


def fit_metrics(model, data_pars, compute_pars, out_pars):
    pass


###############################################################################################################
def plot(net, x, target, backcast_length, forecast_length, grad_step, out_path="./"):
    import matplotlib.pyplot as plt

    net.eval()
    _, f = net(torch.tensor(x, dtype=torch.float))
    subplots = [221, 222, 223, 224]

    plt.figure(1)
    plt.subplots_adjust(top=0.88)
    for i in range(4):
        ff, xx, yy = f.cpu().numpy()[i], x[i], target[i]
        plt.subplot(subplots[i])
        plt.plot(range(0, backcast_length), xx, color="b")
        plt.plot(
            range(backcast_length, backcast_length + forecast_length), yy, color="g"
        )
        plt.plot(
            range(backcast_length, backcast_length + forecast_length), ff, color="r"
        )
        # plt.title(f'step #{grad_step} ({i})')

    output = f"{out_path}/n_beats_{grad_step}.png"
    plt.savefig(output)
    plt.clf()
    print("Saved image to {}.".format(output))


def plot_model(net, x, target, grad_step, data_pars, disable_plot=False):
    forecast_length = data_pars["input_pars"]["forecast_length"]
    backcast_length = data_pars["input_pars"]["backcast_length"]

    # batch_size = compute_pars["batch_size"]  # greater than 4 for viz
    # disable_plot = compute_pars.get("disable_plot", False)

    if not disable_plot:
        print("plot()")
        plot(net, x, target, backcast_length, forecast_length, grad_step)


def plot_predict(x_test, y_test, p, data_pars, compute_pars, out_pars):
    import matplotlib.pyplot as plt

    forecast_length = data_pars["forecast_length"]
    backcast_length = data_pars["backcast_length"]
    norm_constant = compute_pars["norm_contsant"]
    out_path = out_pars["out_path"]
    output = f"{out_path}/n_beats_test.png"

    subplots = [221, 222, 223, 224]
    plt.figure(1)
    plt.subplots_adjust(top=0.88)
    for plot_id, i in enumerate(np.random.choice(range(len(p)), size=4, replace=False)):
        ff, xx, yy = (
            p[i] * norm_constant,
            x_test[i] * norm_constant,
            y_test[i] * norm_constant,
        )
        plt.subplot(subplots[plot_id])
        plt.grid()
        plt.plot(range(0, backcast_length), xx, color="b")
        plt.plot(
            range(backcast_length, backcast_length + forecast_length), yy, color="g"
        )
        plt.plot(
            range(backcast_length, backcast_length + forecast_length), ff, color="r"
        )
    plt.savefig(output)
    plt.clf()
    print("Saved image to {}.".format(output))


###############################################################################################################
# save and load model helper function


def save_checkpoint(model, optimiser, grad_step, CHECKPOINT_NAME="mycheckpoint"):
    torch.save(
        {
            "grad_step": grad_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
        },
        CHECKPOINT_NAME,
    )


def load_checkpoint(model, optimiser, CHECKPOINT_NAME="nbeats-fiting-checkpoint.th"):
    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        grad_step = checkpoint["grad_step"]
        print(f"Restored checkpoint from {CHECKPOINT_NAME}.")
        return grad_step
    return 0


def save(model, session, save_pars):
    optimiser = session
    grad_step = save_pars["grad_step"]
    CHECKPOINT_NAME = save_pars["checkpoint_name"]
    torch.save(
        {
            "grad_step": grad_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
        },
        CHECKPOINT_NAME,
    )


# def load(model, optimiser, CHECKPOINT_NAME='nbeats-fiting-checkpoint.th'):
def load(load_pars):
    model = None
    session = None

    CHECKPOINT_NAME = load_pars["checkpoint_name"]
    optimiser = session

    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        grad_step = checkpoint["grad_step"]
        print(f"Restored checkpoint from {CHECKPOINT_NAME}.")
        return grad_step
    return 0


#############################################################################################################
def get_params(param_pars, **kw):
    import json

    pp = param_pars
    choice = pp["choice"]
    config_mode = pp["config_mode"]
    data_path = pp["data_path"]

    if choice == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, mode="r"))
        cf = cf[config_mode]
        return cf["model_pars"], cf["data_pars"], cf["compute_pars"], cf["out_pars"]

    if choice == "test01":
        log(
            "#### Path params   ########################################################"
        )
        data_path = path_norm("dataset/timeseries/milk.csv")
        out_path = path_norm("ztest/model_tch/nbeats/")
        model_path = os.path.join(out_path, "model")
        print(data_path, out_path)

        data_pars = {
            "data_path": data_path,
            "forecast_length": 5,
            "backcast_length": 10,
        }

        device = torch.device("cpu")

        model_pars = {
            "stack_types": [NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
            "device": device,
            "nb_blocks_per_stack": 3,
            "forecast_length": 5,
            "backcast_length": 10,
            "thetas_dims": [7, 8],
            "share_weights_in_stack": False,
            "hidden_layer_units": 256,
        }

        compute_pars = {
            "batch_size": 100,
            "disable_plot": False,
            "norm_contsant": 1.0,
            "result_path": "n_beats_test{}.png",
            "model_path": "mycheckpoint",
        }

        out_pars = {
            "out_path": out_path + "/",
            "model_checkpoint": out_path + "/model_checkpoint/",
        }

        return model_pars, data_pars, compute_pars, out_pars


#############################################################################################################


def test(data_path="dataset/milk.csv"):
    ###loading the command line arguments

    log("#### Loading params   #######################################")
    param_pars = {"choice": "test01", "data_path": "dataset/", "config_mode": "test01"}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)

    log("#### Loading dataset  #######################################")
    x_train, x_test, y_train, y_test = get_dataset(**data_pars)

    log("#### Model setup   ##########################################")
    model = NBeatsNet(**model_pars)

    log("#### Model fit   ############################################")
    model, optimiser = fit(model, data_pars, compute_pars, out_pars)

    log("#### Predict    #############################################")
    ypred = predict(model, data_pars, compute_pars, out_pars)
    print(ypred)

    log("#### Plot     ###############################################")
    plot_predict(x_test, y_test, ypred, data_pars, compute_pars, out_pars)


if __name__ == "__main__":
    VERBOSE = True
    test()
