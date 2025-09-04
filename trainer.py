import sys
import logging
import copy

import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np


def train(args):

    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    args["local_time"] = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    # get GPU model names
    gpu_names = []
    for dev in device:
        if dev.isdigit():  # check if it is a GPU device number
            idx = int(dev)
            if idx < torch.cuda.device_count():
                gpu_names.append(torch.cuda.get_device_name(idx))
    args['gpu_models'] = gpu_names  # add GPU model list to args

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(
        args["model_name"], args["dataset"], init_cls, args["increment"]
    )

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )

    args["nb_classes"] = data_manager.nb_classes  # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve, route_curve, lora_expert_curve = (
        {"top1": [], "top5": []},
        {"top1": [], "top5": []},
        {"top1": [], "top5": []},
        {"top1": [], "top5": []},
    )
    cnn_matrix = []
    total_train_time = 0.0
    total_test_time = 0.0
    for task in range(data_manager.nb_tasks):

        total_params = sum(p.numel() for p in model._network.backbone.parameters())
        total_trainable_params = sum(
            p.numel() for p in model._network.backbone.parameters() if p.requires_grad
        )
        logging.info("All params: {}".format(total_params))
        logging.info("Trainable params: {}".format(total_trainable_params))
        # if total_params != total_trainable_params:
        #     for name, param in model._network.backbone.named_parameters():
        #         if param.requires_grad:
        #             logging.info("{}: {}".format(name, param.numel()))

        # Train model incrementally
        model.incremental_train(data_manager)

        # Evaluate model
        cnn_accy = model._eval()

        model.after_task()

        cnn_keys = [key for key in cnn_accy["grouped"].keys() if "-" in key]
        cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
        cnn_matrix.append(cnn_values)

        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])

        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info(
            "Average Accuracy (CNN): {:.2f}".format(
                sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
            )
        )
    print(f"\n{'=' * 80}")
    print(
        "Finished {}_init{}_inc{}: {}  ".format(
            args["dataset"],
            args["init_cls"],
            args["increment"],
            args["backbone_type"],
        )
    )
    print("Base Accuracy: {}".format(round(cnn_curve["top1"][0], 2)))
    print("Lase Accuracy: {}".format(round(cnn_curve["top1"][-1], 2)))
    print("Average Accuracy (Top1): {}".format(round(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), 2)))
    print("PD: {:.2f}".format(cnn_curve["top1"][0] - cnn_curve["top1"][-1]))
    print("Backward Transfer (BWT):", backward_transfer(np.array(cnn_matrix)))
    print("Forward Transfer (FWT):", forward_transfer(args["dataset"], np.array(cnn_matrix)))
    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        print("Accuracy Matrix (CNN):")
        print(np_acctable)

    print(f"{'=' * 80}\n")


def _set_device(args):
    """
    Set device (CPU/GPU) for training.
    """
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

def backward_transfer(matrix):
    """
    Calculate the backward transfer from the accuracy matrix.
    """

    # --- Backward Transfer (BWT) ---
    # BWT = (1/(T-1)) * Σ(i=1 to T-1) (R_T,i - R_i,i)
    # R_T,i: after learning the last task T, the accuracy on task i -> corresponds to the last row of the matrix
    # R_i,i: just after learning task i, the accuracy on task i -> corresponds to the diagonal of the matrix

    # R_T,i (i=1..T-1) -> matrix[-1, 0:T-1]
    # R_i,i (i=1..T-1) -> np.diag(matrix)[0:T-1]

    last_row_accs = matrix[-1, :-1]  # the accuracies of all tasks after learning the last task T
    diagonal_accs = np.diag(matrix)[:-1]  # the accuracies of all tasks just after learning themselves

    bwt_diffs = last_row_accs - diagonal_accs
    backward_transfer = np.mean(bwt_diffs)

    return np.round(backward_transfer, 2)

def forward_transfer(dataset, matrix):
    """
    Calculate the forward transfer from the accuracy matrix.
    """
    if dataset == "cub":
        rand_init_acc = np.array([86.51, 52.53, 65.04, 67.86, 74.62, 57.89, 71.82, 87.97, 68.6, 83.61, 85.37])
    elif dataset == "cifar224":
        rand_init_acc = np.array([77.5, 44.4, 51.8, 30.4, 56.8, 68.4, 44.0, 36.4, 41.4])
    elif dataset == "mini_imagenet":
        rand_init_acc = np.array([94.97, 90.6, 69.2, 81.2, 82.8, 73.4, 70.2, 90.6, 89.8])
    else:
        rand_init_acc = np.array([62.95, 28.08, 51.15, 38.64, 56.76, 50.93, 41.38, 61.25, 49.1, 43.37, 44.44]) # INR

    fwt_accs = np.diag(matrix, k=1)  # get R_0,1, R_1,2, ... (0-based)
    rand_init_acc_for_fwt = rand_init_acc[1:]  # get R_0,1, R_0,2, ... (0-based)

    fwt_diffs = fwt_accs - rand_init_acc_for_fwt
    forward_transfer = np.mean(fwt_diffs)

    return np.round(forward_transfer, 2)
