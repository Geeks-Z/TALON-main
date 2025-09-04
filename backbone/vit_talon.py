# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
from sympy import false
from timm.models.layers import DropPath
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import timm
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model

import logging
import os
from collections import OrderedDict
import torch
import copy


class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=768,
                 rank=None, ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = config.attn_bn if rank is None else rank

        self.down_proj = nn.Linear(self.n_embd, self.down_size, bias=False)
        self.up_proj = nn.Linear(self.down_size, self.n_embd, bias=False)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            # nn.init.zeros_(self.down_proj.bias)
            # nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        # residual = x if residual is None else residual

        down = self.down_proj(x)
        up = self.up_proj(down)

        # if add_residual:
        #     output = up + residual
        # else:
        #     output = up

        return up  # output


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x, q_lora=None, k_lora=None, v_lora=None):
        B, N, C = x.shape

        q = self.q_proj(x) + (q_lora(x) if q_lora is not None else torch.zeros_like(self.q_proj(x)))
        k = self.k_proj(x) + (k_lora(x) if k_lora is not None else torch.zeros_like(self.k_proj(x)))
        v = self.v_proj(x) + (v_lora(x) if v_lora is not None else torch.zeros_like(self.v_proj(x)))

        k = self._shape(k, -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(v, -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x, q_lora, k_lora, v_lora, mlp_lora):
        atten_output = self.attn(self.norm1(x), q_lora, k_lora, v_lora)
        x = x + self.drop_path(atten_output)
        if mlp_lora is not None:
            adapt_x = mlp_lora(x, add_residual=False)
        else:
            adapt_x = None

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))
        if adapt_x is not None:
            x = x + adapt_x

        x = residual + x
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None):
        super().__init__()
        print("I'm using ViT with LoRA-Expert.")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # self.init_weights(weight_init)

        ######### MAE begins ############
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.config = tuning_config
        self._device = tuning_config._device

        self.t_mlp_lora_list = nn.ModuleList()  # t_ teacher lora
        self.t_q_lora_list = nn.ModuleList()
        self.t_k_lora_list = nn.ModuleList()
        self.t_v_lora_list = nn.ModuleList()

        self.cur_t_q_lora = None
        self.cur_t_k_lora = None
        self.cur_t_v_lora = None
        self.cur_t_mlp_lora = None

        self.cur_stu_lora = None
        self.cur_stu_k_lora = None
        self.cur_stu_v_lora = None
        self.cur_stu_mlp_lora = None

        self.init_lora()

    def init_lora(self):
        config = self.config
        t_lora_position = config['t_lora_positions']
        # s_lora_position = config['s_lora_positions']
        if 'mlp' in t_lora_position:
            self.cur_t_mlp_lora = nn.ModuleList()
            for i in range(len(self.blocks)):
                t_mlp_lora = Adapter(self.config, d_model=self.embed_dim, rank=config.t_rank, ).to(self._device)
                self.cur_t_mlp_lora.append(t_mlp_lora)
            self.cur_t_mlp_lora.requires_grad_(True)

        if 'q' in t_lora_position:
            self.cur_t_q_lora = nn.ModuleList()
            for i in range(len(self.blocks)):
                t_q_lora = Adapter(self.config, d_model=self.embed_dim, rank=config.t_rank, ).to(self._device)
                self.cur_t_q_lora.append(t_q_lora)
            self.cur_t_q_lora.requires_grad_(True)

            # self.cur_stu_lora = nn.ModuleList()

        if 'k' in t_lora_position:
            self.cur_t_k_lora = nn.ModuleList()
            for i in range(len(self.blocks)):
                t_k_lora = Adapter(self.config, d_model=self.embed_dim, rank=config.t_rank, ).to(self._device)
                self.cur_t_k_lora.append(t_k_lora)
            self.cur_t_k_lora.requires_grad_(True)

            # self.cur_stu_k_lora = nn.ModuleList()

        if 'v' in t_lora_position:
            self.cur_t_v_lora = nn.ModuleList()
            for i in range(len(self.blocks)):
                t_v_lora = Adapter(self.config, d_model=self.embed_dim, rank=config.t_rank, ).to(self._device)
                self.cur_t_v_lora.append(t_v_lora)
            self.cur_t_v_lora.requires_grad_(True)

            # self.cur_stu_v_lora = nn.ModuleList()

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for i in range(len(self.blocks)):
            self.cur_adapter[i].requires_grad = True

    def adapter_update(self):
        # only update teacher lora
        if self.cur_t_q_lora is not None:
            self.t_q_lora_list.append(copy.deepcopy(self.cur_t_q_lora))
        if self.cur_t_k_lora is not None:
            self.t_k_lora_list.append(copy.deepcopy(self.cur_t_k_lora))
        if self.cur_t_v_lora is not None:
            self.t_v_lora_list.append(copy.deepcopy(self.cur_t_v_lora))
        if self.cur_t_mlp_lora is not None:
            self.t_mlp_lora_list.append(copy.deepcopy(self.cur_t_mlp_lora))
        self.init_lora()

    def forward_features_student_adapter(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for layer_idx, blk in enumerate(self.blocks):
            stu_q_lora = self.cur_stu_lora[layer_idx] if 'q' in self.config.s_lora_positions else None
            stu_k_lora = self.cur_stu_k_lora[layer_idx] if 'k' in self.config.s_lora_positions else None
            stu_v_lora = self.cur_stu_v_lora[layer_idx] if 'v' in self.config.s_lora_positions else None
            stu_mlp_lora = self.cur_stu_mlp_lora[layer_idx] if 'mlp' in self.config.s_lora_positions else None
            x = blk(x, stu_q_lora, stu_k_lora, stu_v_lora, stu_mlp_lora)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        res = dict()
        res['features'] = outcome
        res['logits'] = self.head(outcome)
        return res

    def forward_features(self, x, adapter_id, train):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if adapter_id == -1:
            x = self.blocks(x)
        else:
            for layer_idx, blk in enumerate(self.blocks):
                if train:
                    t_q_lora = self.cur_t_q_lora[layer_idx] if 'q' in self.config.t_lora_positions else None
                    t_k_lora = self.cur_t_k_lora[layer_idx] if 'k' in self.config.t_lora_positions else None
                    t_v_lora = self.cur_t_v_lora[layer_idx] if 'v' in self.config.t_lora_positions else None
                    t_mlp_lora = self.cur_t_mlp_lora[layer_idx] if 'mlp' in self.config.t_lora_positions else None
                else:
                    list_len = next(
                        (len(lst) for lst in
                         [self.t_q_lora_list, self.t_k_lora_list, self.t_v_lora_list, self.t_mlp_lora_list]
                         if lst),
                        0
                    )
                    if adapter_id == list_len:
                        t_q_lora = self.cur_t_q_lora[layer_idx] if 'q' in self.config.t_lora_positions else None
                        t_k_lora = self.cur_t_k_lora[layer_idx] if 'k' in self.config.t_lora_positions else None
                        t_v_lora = self.cur_t_v_lora[layer_idx] if 'v' in self.config.t_lora_positions else None
                        t_mlp_lora = self.cur_t_mlp_lora[layer_idx] if 'mlp' in self.config.t_lora_positions else None

                    elif adapter_id < list_len:
                        t_q_lora = self.t_q_lora_list[adapter_id][
                            layer_idx] if 'q' in self.config.t_lora_positions else None
                        t_k_lora = self.t_k_lora_list[adapter_id][
                            layer_idx] if 'k' in self.config.t_lora_positions else None
                        t_v_lora = self.t_v_lora_list[adapter_id][
                            layer_idx] if 'v' in self.config.t_lora_positions else None
                        t_mlp_lora = self.t_mlp_lora_list[adapter_id][
                            layer_idx] if 'mlp' in self.config.t_lora_positions else None
                    else:
                        raise ValueError("adapter_id is wrong.")
                x = blk(x, t_q_lora, t_k_lora, t_v_lora, t_mlp_lora)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        res = dict()
        res['x'] = outcome
        res['pre_logits'] = outcome
        res['features'] = outcome

        return res

    def forward_head(self, res):
        x = res['x']

        res['logits'] = self.head(x)
        return res

    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        if fc_only:
            res = dict()
            res['logits'] = self.head(x)
            return res

        res = self.forward_features(x, adapter_id, train)
        res = self.forward_head(res)

        return res


def vit_base_patch16_224_talon(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=kwargs['num_classes'])
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768 * 2]
            v_weight = qkv_weight[768 * 2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768 * 2]
            v_bias = qkv_bias[768 * 2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)

    # freeze all but the adapter and head
    for name, p in model.named_parameters():
        if 'head' in name or 'lora' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


def vit_base_patch16_224_in21k_talon(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True,
                                         num_classes=kwargs['num_classes'])
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768 * 2]
            v_weight = qkv_weight[768 * 2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768 * 2]
            v_bias = qkv_bias[768 * 2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)

    # freeze all but the adapter and head
    for name, p in model.named_parameters():
        if 'head' in name or 'lora' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


def vit_large_patch16_224_talon(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    checkpoint_model = timm.create_model("vit_large_patch16_224", pretrained=True, num_classes=kwargs['num_classes'])
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:1024]
            k_weight = qkv_weight[1024:1024 * 2]
            v_weight = qkv_weight[1024 * 2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:1024]
            k_bias = qkv_bias[1024:1024 * 2]
            v_bias = qkv_bias[1024 * 2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)

    # freeze all but the adapter and head
    for name, p in model.named_parameters():
        if 'head' in name or 'lora' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


def vit_base_patch16_224_dino_talon(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    checkpoint_model = timm.create_model("vit_base_patch16_224_dino", pretrained=True,
                                         num_classes=kwargs['num_classes'])
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768 * 2]
            v_weight = qkv_weight[768 * 2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768 * 2]
            v_bias = qkv_bias[768 * 2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)

    # freeze all but the adapter and head
    for name, p in model.named_parameters():
        if 'head' in name or 'lora' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


def vit_base_patch16_224_sam_talon(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    checkpoint_model = timm.create_model("vit_base_patch16_224_sam", pretrained=True,
                                         num_classes=kwargs['num_classes'])
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768 * 2]
            v_weight = qkv_weight[768 * 2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768 * 2]
            v_bias = qkv_bias[768 * 2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)

    # freeze all but the adapter and head
    for name, p in model.named_parameters():
        if 'head' in name or 'lora' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model
