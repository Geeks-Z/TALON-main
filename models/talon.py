import copy
import logging
import os
import random

import numpy as np
import torch
import wandb
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import TALONNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
import timm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler as DS
from pytorch_lightning.loggers import WandbLogger

num_workers = 16
EPSILON = 1e-8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = TALONNet(args, True)
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()
        self.knn_feature2cls = dict()

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.kd_student_lr = args["kd_student_lr"]
        self.kd_student_epochs = args["kd_student_epochs"]
        self.weight_decay = (
            args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        )
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args

        for n, p in self._network.backbone.named_parameters():
            if "lora" not in n and "head" not in n:
                p.requires_grad = False
        self.teacher_cls_prototype = []  # teacher

        # total_params = sum(p.numel() for p in self._network.backbone.parameters())
        # logging.info(f'{total_params:,} model total parameters.')
        # total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        # logging.info(f'{total_trainable_params:,} model training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        # if total_params != total_trainable_params:
        #     for name, param in self._network.backbone.named_parameters():
        #         if param.requires_grad:
        #             logging.info("{}: {}".format(name, param.numel()))

    def replace_fc(self, loader):
        model = self._network.to(self._device)
        model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.backbone.forward_features_student_adapter(data)[
                    "features"
                ]
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(loader.dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto
        return model

    def after_task(self):
        self._known_classes = self._total_classes
        # self.frozen_teacher_list_parameters()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

        if self._cur_task != 0:
            self.batch_size = self.args["fs_batch_size"]

        for i in range(self._known_classes, self._total_classes):
            self.cls2task[i] = self._cur_task

        self._network.update_fc(self._total_classes)

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # Set up data loaders
        # logging.info("Selecting samples for train_dataset.")
        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            kshot=self.args["kshot"],
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # logging.info("selecting samples for future_dataset.")

        if self._total_classes < self.args['nb_classes']:
            self.future_dataset = data_manager.get_dataset(
                np.arange(self._total_classes, self.args["nb_classes"]),
                source="train",
                mode="train",
                kshot=self.args["kshot"],
            )
            self.future_loader = DataLoader(
                self.future_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        # Prototypical network training data
        # logging.info("Selecting samples for protonet.")
        self.train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
            kshot=self.args["kshot"],
        )
        self.train_loader_for_protonet = DataLoader(
            self.train_dataset_for_protonet,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self._train(self.train_loader, self.test_loader)

    def _train(self, train_loader, test_loader):
        self._network.backbone.to(self._device)

        # optimizer_init（ cur and base_fc）
        optimizer = self.get_optimizer(self._network.backbone, mode="init")
        scheduler = self.get_scheduler(optimizer)

        self._init_train(train_loader, optimizer, scheduler)
        self._update_prototype()

        # Distillation phase optimizer (only optimize stu)
        student_optimizer = self.get_optimizer(self._network.backbone, mode="kd")
        student_scheduler = self.get_scheduler(student_optimizer)

        # Update class prototype for the student
        self.update_cls_prototype(train_loader)
        self.kd_student_train(self.train_loader, student_optimizer, student_scheduler)
        # update student classifier weights (known_classes -> total_classes)
        self.replace_fc(self.train_loader_for_protonet)
        self._network.backbone.adapter_update()

    def get_optimizer(self, model, mode):
        param_groups = []

        if mode == "init":
            cur_params = [
                p
                for name, p in model.named_parameters()
                if "cur" in name and p.requires_grad
            ]
            base_fc_params = [
                p
                for name, p in model.named_parameters()
                if "lora" not in name and p.requires_grad
            ]
            param_groups.extend(
                [
                    {
                        "params": cur_params,
                        "lr": self.init_lr,
                        "weight_decay": self.weight_decay,
                    },
                    {
                        "params": base_fc_params,
                        "lr": self.init_lr * 0.1,
                        "weight_decay": self.weight_decay,
                    },
                ]
            )
        elif mode == "kd":
            student_params = [
                p
                for name, p in model.named_parameters()
                if "stu_" in name and p.requires_grad
            ]
            param_groups.append(
                {
                    "params": student_params,
                    "lr": self.kd_student_lr,
                    "weight_decay": self.weight_decay,
                }
            )
        else:
            raise ValueError(f"Invalid optimizer mode: {mode}")

        optimizer_type = self.args["optimizer"]
        if optimizer_type == "sgd":
            optimizer = optim.SGD(param_groups, momentum=0.9)
        elif optimizer_type == "adam":
            optimizer = optim.Adam(param_groups)
        elif optimizer_type == "adamw":
            optimizer = optim.AdamW(param_groups)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["tuned_epoch"], eta_min=self.min_lr
            )
        elif self.args["scheduler"] == "steplr":
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["init_milestones"],
                gamma=self.args["init_lr_decay"],
            )
        elif self.args["scheduler"] == "constant":
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, optimizer, scheduler):

        prog_bar = tqdm(range(self.args["tuned_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                output = self._network(inputs, adapter_id=self._cur_task, train=True)
                logits = output["logits"][:, : self._total_classes]
                logits[:, : self._known_classes] = float("-inf")

                # 分类loss
                loss = F.cross_entropy(logits, targets.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args["tuned_epoch"],
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def kd_student_train(self, train_loader, optimizer, scheduler):
        if self._cur_task == 0:
            if self._network.backbone.cur_t_q_lora is not None:
                self._network.backbone.cur_stu_lora = copy.deepcopy(
                    self._network.backbone.cur_t_q_lora
                )
            if self._network.backbone.cur_t_k_lora is not None:
                self._network.backbone.cur_stu_k_lora = copy.deepcopy(
                    self._network.backbone.cur_t_k_lora
                )
            if self._network.backbone.cur_t_v_lora is not None:
                self._network.backbone.cur_stu_v_lora = copy.deepcopy(
                    self._network.backbone.cur_t_v_lora
                )
            if self._network.backbone.cur_t_mlp_lora is not None:
                self._network.backbone.cur_stu_mlp_lora = copy.deepcopy(
                    self._network.backbone.cur_t_mlp_lora
                )

            # self._network.backbone.student_adapter = copy.deepcopy(self._network.backbone.cur_adapter)
            return
        # print(
        #     "session {} kd_student_train_images: {}".format(
        #         self._cur_task, len(train_loader.dataset)
        #     )
        # )
        # print("-" * 100)
        prog_bar = tqdm(range(self.args["kd_student_epochs"]))
        for _, epoch in enumerate(prog_bar):
            losses, cls_loss, kd_loss = 0.0, 0.0, 0.0
            temperature = self.args["temperature"]
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)

                output = self._network.backbone.forward_features_student_adapter(inputs)
                cos_similarity = self.cal_cos_sim()
                student_logits = output["logits"][:, : self._total_classes]

                # logtis distillation
                # teacher_logits = torch.zeros(
                #     len(inputs), self._total_classes, device=self._device
                # )
                # for t_id in range(self._cur_task + 1):
                #     with torch.no_grad():
                #         t_id_teacher_logits = self._network.backbone(
                #             inputs, adapter_id=t_id, train=False
                #         )["logits"][:, : self._total_classes]
                #     teacher_logits += t_id_teacher_logits * cos_similarity[t_id]

                # L2 loss
                # kd_loss = F.mse_loss(student_logits, teacher_logits)

                # KL divergence loss for logits distillation
                # student_probs = F.log_softmax(student_logits, dim=1)
                # teacher_probs = F.softmax(teacher_logits.detach(), dim=1)
                # kd_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean")

                # feature distillation
                student_embeddings = output["features"]
                teacher_embeddings = torch.zeros(
                    len(inputs), self._network.backbone.out_dim, device=self._device
                )
                for t_id in range(self._cur_task + 1):
                    with torch.no_grad():
                        t_id_teacher_embeddings = self._network.backbone(inputs, adapter_id=t_id, train=False)[
                            "features"]
                    teacher_embeddings += t_id_teacher_embeddings * cos_similarity[t_id]
                # kd_loss = 1 - F.cosine_similarity(student_embeddings, teacher_embeddings, dim=1).mean()
                kd_loss = F.mse_loss(student_embeddings, teacher_embeddings)
                # student_probs = F.softmax(student_embedding / temperature, dim=-1)
                # teacher_probs = F.softmax(teacher_embeddings.detach() / temperature, dim=-1)
                # kd_loss = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')

                # cls loss
                mask = (
                    torch.arange(self._total_classes, device=student_logits.device)
                    < self._known_classes
                )
                student_logits = student_logits.masked_fill(
                    mask.unsqueeze(0), float("-inf")
                )
                cls_loss = F.cross_entropy(student_logits, targets.long())

                loss = cls_loss + kd_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()

            if scheduler:
                scheduler.step()
            # train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "KD Student {}, Epoch {}/{} => Loss {:.3f}".format(
                self._cur_task,
                epoch + 1,
                self.args["kd_student_epochs"],
                losses / len(train_loader),
            )
            prog_bar.set_description(info)
        # wandb.watch_called = False
        logging.info(info)

    def cal_cos_sim(self):
        cos_similarity = []
        for task_id in range(self._cur_task + 1):
            cur_cls_prototype = self.cal_cls_prototype(self.train_loader, task_id)
            cur_cls_prototype_tensor = torch.cat(cur_cls_prototype, dim=0)
            # count = sum(v == task_id for v in self.cls2task.values())
            first, last = None, None
            for k, v in self.cls2task.items():
                if v == task_id:
                    if first is None:
                        first = k
                    last = k
            # old_cls_prototype = self.teacher_cls_prototype[first:last+1]
            similarity = nn.functional.cosine_similarity(
                cur_cls_prototype_tensor.detach().unsqueeze(1),
                self.teacher_cls_prototype[first : last + 1].detach().unsqueeze(0),
                dim=2,
            )
            cos_similarity.append(torch.sum(torch.sum(similarity, dim=1), dim=0))
        cos_similarity = torch.stack(cos_similarity, dim=0)
        temperature = 10.0  # 调节温度系数
        scaled_input = cos_similarity / temperature
        softmax_result = F.softmax(scaled_input, dim=0)
        return softmax_result

    # update cls protot
    @torch.no_grad()
    def _update_prototype(self):

        cls_prototype = self.cal_cls_prototype(self.train_loader, self._cur_task)
        # update cls_prototype
        cls_prototype_tensor = torch.cat(cls_prototype, dim=0)
        if len(self.teacher_cls_prototype) == 0:
            self.teacher_cls_prototype = cls_prototype_tensor.to(self._device)
        else:
            self.teacher_cls_prototype = torch.cat(
                (self.teacher_cls_prototype, cls_prototype_tensor), dim=0
            ).to(self._device)

    @torch.no_grad()
    def cal_cls_prototype(self, data_loader, lora_id):
        model = self._network.to(self._device)
        model.eval()
        embedding_list = []
        label_list = []
        cls_prototype = []
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model(data, adapter_id=lora_id, train=False)["features"]
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0).to(self._device).clone().detach()
            cls_prototype.append(proto.unsqueeze(0))
        return cls_prototype

    def frozen_teacher_list_parameters(self):
        for name, param in self._network.backbone.named_parameters():
            if "list" in name and param.requires_grad:
                param.requires_grad = False

    def _eval_cnn(self, loader, y_pred, y_true):
        for _, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            with torch.no_grad():
                features = self._network.backbone.forward_features_student_adapter(
                    inputs
                )["features"]
                outputs = self._network.fc(features)["logits"][:, : self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        # return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_future_cnn(self, loader, y_pred, y_true):

        if self._total_classes < self.args['nb_classes']:
            self.update_future_head(self._network, self.future_loader)
            for _, (_, inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                with torch.no_grad():
                    features = self._network.backbone.forward_features_student_adapter(
                        inputs)["features"]
                    outputs = self._network.future_head(features)["logits"]
                predicts = torch.topk(
                    outputs, k=self.topk, dim=1, largest=True, sorted=True
                )[
                    1
                ]  # [bs, topk]
                y_pred.append(predicts.cpu().numpy())  # Adjust for future tasks
                y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def update_future_head(self, model, future_loader):
        # for i in range(self._total_classes):
        #     model.future_head.weight.data[i] = model.fc.weight.data[i]
        model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(future_loader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.backbone.forward_features_student_adapter(data)[
                    "features"
                ]
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(future_loader.dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            model.future_head.weight.data[class_index] = proto
