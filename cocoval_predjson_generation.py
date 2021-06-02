# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from models import build_model as build_yolos_model
from datasets import build_dataset, get_coco_api_from_dataset
# from timm.scheduler import create_scheduler
# from new_models import build_model
from util.scheduler import create_scheduler
from datasets.coco_eval import CocoEvaluator

from util import box_ops
import torch.nn.functional as F
class MyPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = box_ops.box_xyxy_to_cxcywh(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        ones_h = torch.ones_like(img_h, device=img_h.device)
        ones_w = torch.ones_like(img_w, device=img_w.device)
        scale_fct = torch.stack([ones_w, ones_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

@torch.no_grad()
def get_val_json(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args):
    model.eval()
    criterion.eval()

    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    jdict = []
    postprocess = MyPostProcess()
    # import pdb;pdb.set_trace()
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocess(outputs, orig_target_sizes)
        imgs_h, imgs_w = orig_target_sizes.unbind(1)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        img_indx=0
        for res_k, res_v in res.items():
            image_id = int(res_k)
            # for k, v in res_v.items():
            scores = res_v['scores'].cpu().tolist()
            labels = res_v['labels'].cpu().tolist()
            boxes = res_v['boxes']
            boxes = boxes.cpu().tolist() #normedcx, normedcy, w, h
            for score, label, box in zip(scores, labels, boxes):
                jdict.append({'image_id': image_id,
                                'category_id': int(label),
                                'bbox': [round(x, 3) for x in box],
                                'score': round(score, 5)})
    output_json = os.path.join(output_dir, args.resume+ "-eval-"+str(args.eval_size)+"eval-pred.json")
    with open(output_json, 'w') as f:
        json.dump(jdict, f)

        # for target, output in zip(targets, results):
            # jdict
    print("%s done" % output_json)
    return

def get_args_parser():
    parser = argparse.ArgumentParser('Set YOLOS', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--eval_size', default=800, type=int)
    
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    # scheduler
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='warmupcos', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step", options:"step", "warmupcos"')
    ## step
    parser.add_argument('--lr_drop', default=100, type=int)  
    ## warmupcosine

    # parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
    #                     help='learning rate noise on/off epoch percentages')
    # parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
    #                     help='learning rate noise limit percent (default: 0.67)')
    # parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
    #                     help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # * model setting
    parser.add_argument("--det_token_num", default=100, type=int,
                        help="Number of det token in the deit backbone")
    parser.add_argument('--backbone_name', default='tiny', type=str,
                        help="Name of the deit backbone to use")
    parser.add_argument('--pre_trained', default='',
                        help="set imagenet pretrained model path if not train yolos from scatch")
    parser.add_argument('--init_pe_size', nargs='+', type=int,
                        help="init pe size (h,w)")
    parser.add_argument('--mid_pe_size', nargs='+', type=int,
                        help="mid pe size (h,w)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients

    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # import pdb;pdb.set_trace()
    model, criterion, postprocessors = build_yolos_model(args)
    # model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    def build_optimizer(model, args):
        if hasattr(model.backbone, 'no_weight_decay'):
            skip = model.backbone.no_weight_decay()
        head = []
        backbone_decay = []
        backbone_no_decay = []
        for name, param in model.named_parameters():
            if "backbone" not in name and param.requires_grad:
                head.append(param)
            if "backbone" in name and param.requires_grad:
                if len(param.shape) == 1 or name.endswith(".bias") or name.split('.')[-1] in skip:
                    backbone_no_decay.append(param)
                else:
                    backbone_decay.append(param)
        param_dicts = [
            {"params": head},
            {"params": backbone_no_decay, "weight_decay": 0., "lr": args.lr},
            {"params": backbone_decay, "lr": args.lr},
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
        return optimizer

    optimizer = build_optimizer(model_without_ddp, args)


    lr_scheduler, _ = create_scheduler(args, optimizer)
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    # import pdb;pdb.set_trace()
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)



    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        get_val_json(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args)

        return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get YOLOS pred json file', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)