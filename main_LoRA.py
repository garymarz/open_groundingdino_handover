# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


import torch.jit
script_method1 = torch.jit.script_method
script1 = torch.jit.script
torch.jit.script_method = script_method
torch.jit.script = script


import torch
import torch.nn as nn
import multiprocessing
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from PIL import ImageFile
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import  BestMetricHolder
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch

from groundingdino.util.utils import clean_state_dict
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file','-c' ,default='config/cfg_odvg.py',type=str)
    parser.add_argument('--options',
        default='text_encoder_type=bert-base-uncased',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument("--datasets", type=str, default='data/train.json', help='path to datasets json')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume',default=False, help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path',default='groundingdino_swint_ogc.pth', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    return parser

class LoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.d_model = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)
        q_ba = self.linear_b_q(self.linear_a_q(x))
        v_ba = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :,  :self.d_model] += q_ba # q part
        qkv[:, :,  -self.d_model:] += v_ba # v part
        return qkv

class LoRA_gdswin(nn.Module):
    def __init__(self, model, rank=256):
        super().__init__()
        self.rank = rank
        assert rank > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        self.A_weights = []
        self.B_weights = []
        for param in model.parameters():
            param.requires_grad = False

        for layer in model.backbone[0].layers:
            for blk in layer.blocks:
                w_qkv_linear = blk.attn.qkv
                self.d_model = w_qkv_linear.in_features
                w_a_linear_q = nn.Linear(self.d_model, self.rank, bias=False)
                w_b_linear_q = nn.Linear(self.rank, self.d_model, bias=False)
                w_a_linear_v = nn.Linear(self.d_model, self.rank, bias=False)
                w_b_linear_v = nn.Linear(self.rank, self.d_model, bias=False)
                self.A_weights.append(w_a_linear_q)
                self.B_weights.append(w_b_linear_q)
                self.A_weights.append(w_a_linear_v)
                self.B_weights.append(w_b_linear_v)
                blk.attn.qkv = LoRA_qkv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v
                )
        self.reset_parameters()
        self.lora_model = model
        

    def reset_parameters(self):
        # Initalisation like in the paper
        for w_A in self.A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.B_weights:
            nn.init.zeros_(w_B.weight)

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.lora_A = nn.Linear(original_linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_linear.out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original_linear(x) + self.lora_B(self.lora_A(x))
    

def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def main(args):
    utils.setup_distributed(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        options = {'text_encoder_type': 'bert-base-uncased'}
        cfg.merge_from_dict(options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))

    with open(args.datasets) as f:
        dataset_meta = json.load(f)
    if args.use_coco_eval:
        args.coco_val_path = dataset_meta["val"][0]["anno"]

    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    logger.debug("build model ... ...")
    model, criterion, postprocessors = build_model_main(args)
    
    checkpoint = torch.load('groundingdino_swint_ogc.pth', map_location='cpu')['model']
    from collections import OrderedDict
    _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
    ignorelist = []
    def check_keep(keyname, ignorekeywordlist):
        for keyword in ignorekeywordlist:
            if keyword in keyname:
                ignorelist.append(keyname)
                return False
        return True

#     logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
    _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
    _load_output = model.load_state_dict(_tmp_st, strict=False)
    logger.info(str(_load_output))
    wo_class_error = False
    logger.debug("build model, done.")
    # def apply_lora_to_transformer_encoder_layer(layer, rank):
    #     layer.self_attn.out_proj = LoRALinear(layer.self_attn.output_proj, rank)
    #     layer.linear1 = LoRALinear(layer.linear1, rank)
    #     layer.linear2 = LoRALinear(layer.linear2, rank)
    # 套用在transformer上的LoRA
    # for encoder_layer in model.transformer.encoder.layers:
    #     apply_lora_to_transformer_encoder_layer(encoder_layer, rank=32)
    # 套用swin_trainformer的LoRA
    model = LoRA_gdswin(model, 512)
    model = model.lora_model
    model.to(device)

    model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    #     model._set_static_graph()
    #     model_without_ddp = model.module
    # n_parameters = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
    # logger.info('number of params:'+str(n_parameters))
    # logger.info("params before freezing:\n"+json.dumps({n: p.numel() for n, p in model_lora.named_parameters() if p.requires_grad}, indent=2))

    # param_dicts = get_param_dict(args, model_without_ddp)
    # logger.info("params after freezing:\n"+json.dumps({n: p.numel() for n, p in model_lora.named_parameters() if p.requires_grad}, indent=2))
    # optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
    #                               weight_decay=args.weight_decay)
    
    logger.debug("build dataset ... ...")
    if not args.eval:
        num_of_dataset_train = len(dataset_meta["train"])
        if num_of_dataset_train == 1:
            dataset_train = build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][0])
        else:
            from torch.utils.data import ConcatDataset
            dataset_train_list = []
            for idx in range(len(dataset_meta["train"])):
                dataset_train_list.append(build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][idx]))
            dataset_train = ConcatDataset(dataset_train_list)
        logger.debug("build dataset, done.")
        logger.debug(f'number of training dataset: {num_of_dataset_train}, samples: {len(dataset_train)}')

    # dataset_val = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])

    if args.distributed:
        # sampler_val = DistributedSampler(dataset_val, shuffle=False)
        if not args.eval:
            sampler_train = DistributedSampler(dataset_train)
    else:
        # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if not args.eval:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
    #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # base_ds = get_coco_api_from_dataset(dataset_val)
    # if args.frozen_weights is not None:
    #     checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    #     model_without_ddp.detr.load_state_dict(clean_state_dict(checkpoint['model']),strict=False) 

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)
        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []
        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

    # logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        _load_output = model.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

    print("Start training")
    start_time = time.time()
    # model.enable_input_require_grads()
    # best_map_holder = BestMetricHolder(use_ema=False)
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None))
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                utils.save_on_master(weights, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)

if __name__ == '__main__':
    multiprocessing.freeze_support() # parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    from functools import partial
    notfailing_checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    torch.utils.checkpoint.checkpoint = notfailing_checkpoint
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
