import argparse
from copy import deepcopy
import logging
import os
import pprint
from typing import Optional

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.sam_generator import (
    create_sam_generator,
    _BaseSAMPseudoLabelGenerator,
)
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

parser = argparse.ArgumentParser(
    description='SAGT')
parser.add_argument('--config',             type=str, required=True)
parser.add_argument('--labeled-id-path',    type=str, required=True)
parser.add_argument('--unlabeled-id-path',  type=str, required=True)
parser.add_argument('--save-path',          type=str, required=True)
# SAM (SAM2 / SAM3)
parser.add_argument('--sam-version',        type=str, default='sam3_hf',
                    choices=['sam2', 'sam3_hf', 'sam3_official'],
                    help='SAM version to use: '
                         'sam2 (official SAM2 package), '
                         'sam3_hf (HF Transformers pipeline, recommended), '
                         'sam3_official (official SAM3 Meta package)')
parser.add_argument('--sam-checkpoint',     type=str, default=None,
                    help='Path to SAM checkpoint (.pt). '
                         'Required for sam2 and sam3_official. '
                         'For sam3_hf, checkpoint is auto-downloaded from HF.')
parser.add_argument('--sam-config',         type=str,
                    default='sam2_hiera_large.yaml',
                    help='SAM2 config name (only used for sam2 version).')
parser.add_argument('--sam-model-name',     type=str, default='facebook/sam3',
                    help='HF model name for sam3_hf version.')
parser.add_argument('--sam-text-prompts',   type=str, nargs='*', default=None,
                    help='Text prompts for sam3_official version '
                         '(e.g. "person" "car" "tree").')
parser.add_argument('--sam-mask-dir',       type=str, default=None,
                    help='Pre-generated SAM mask directory. '
                         'If provided, masks are loaded rather than generated online.')
parser.add_argument('--env-csv', type=str, default=None,
                    help='Path to env label CSV (image_name,landslide_exists,'
                         'has_building,has_road,has_water,has_vegetation). '
                         'When provided, adds an auxiliary BCE loss.')
parser.add_argument('--env-loss-weight', type=float, default=0.1,
                    help='Weight λ for env classification loss (default: 0.1).')
parser.add_argument('--consensus-weight', action='store_true', default=False,
                    help='Enable SAM-Teacher consensus weighting: assign '
                         'continuous per-pixel weights based on agreement '
                         'between SAM and Teacher pseudo-labels.')
parser.add_argument('--cw-sam-only', type=float, default=0.7,
                    help='SAGT weight for SAM-only pixels (default: 0.7).')
parser.add_argument('--cw-teacher-only', type=float, default=0.5,
                    help='SAGT weight for Teacher-only pixels (default: 0.5).')
parser.add_argument('--cw-disagree', type=float, default=0.3,
                    help='SAGT weight for disagreement pixels (default: 0.3).')
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port',       default=None, type=int)


MODEL_CONFIGS = {
    'small': {'encoder_size': 'small', 'features': 64,
               'out_channels': [48,   96,   192,  384]},
    'base':  {'encoder_size': 'base',  'features': 128,
               'out_channels': [96,   192,  384,  768]},
    'large': {'encoder_size': 'large', 'features': 256,
               'out_channels': [256,  512,  1024, 1024]},
    'giant': {'encoder_size': 'giant', 'features': 384,
               'out_channels': [1536, 1536, 1536, 1536]},
}

EMBED_DIMS = {'small': 384, 'base': 768, 'large': 1024, 'giant': 1536}


def _adapt_dinov3_weights(state_dict):
    """Adapt official DINOv3 weights to match our DinoV3VisionTransformer."""
    new_dict = {}
    renamed = []
    skipped = []

    for k, v in state_dict.items():
        new_k = k

        if k == 'storage_tokens':
            new_k = 'register_tokens'
            renamed.append((k, new_k))
        elif k.startswith('rope_embed.'):
            skipped.append(k)
            continue
        elif k.endswith('.qkv.bias_mask'):
            skipped.append(k)
            continue
        elif k.endswith('.attn.qkv.bias_mask'):
            skipped.append(k)
            continue

        new_dict[new_k] = v

    if renamed:
        print(f'  [DINOv3 weights] Renamed {len(renamed)} keys:')
        for old, new in renamed[:5]:
            print(f'    {old} → {new}')
    if skipped:
        print(f'  [DINOv3 weights] Skipped {len(skipped)} keys:')
        for k in skipped[:5]:
            print(f'    {k}')

    return new_dict


def _build_model(cfg, num_env_classes=None):
    backbone_parts = cfg['backbone'].split('_')          # e.g. ['dinov2', 'small']
    encoder_type = backbone_parts[0]                       # 'dinov2' or 'dinov3'
    size_key = backbone_parts[-1]                           # e.g. 'small'
    model = DPT(**{**MODEL_CONFIGS[size_key], 'nclass': cfg['nclass'],
                    'encoder_type': encoder_type,
                    'num_env_classes': num_env_classes})

    weight_path = f'./pretrained/{cfg["backbone"]}.pth'
    if not os.path.exists(weight_path):
        if encoder_type == 'dinov3':
            raise FileNotFoundError(
                f'\n{"="*60}\n'
                f'  Missing DINOv3 pretrained weight: {weight_path}\n'
                f'  {"="*60}\n'
                f'  Please download DINOv3 weights using ONE of these methods:\n\n'
                f'  Method 1 (HuggingFace, recommended):\n'
                f'    1. Request access at: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m\n'
                f'    2. Login: huggingface-cli login\n'
                f'    3. Run: python scripts/download_dinov3_weights.py --model {size_key}\n\n'
                f'  Method 2 (Meta official):\n'
                f'    1. Request access at: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/\n'
                f'    2. Download the ViT-S/16 or ViT-B/16 backbone checkpoint\n'
                f'    3. Rename to: pretrained/{cfg["backbone"]}.pth\n'
                f'{"="*60}\n'
            )
        else:
            raise FileNotFoundError(
                f'Missing pretrained weight: {weight_path}. '
                f'Please download DINOv2 {size_key} checkpoint and place it there.'
            )

    state = torch.load(weight_path, map_location='cpu')

    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    elif isinstance(state, dict) and 'model' in state:
        state = state['model']

    if encoder_type == 'dinov3':
        state = _adapt_dinov3_weights(state)

    missing, unexpected = model.backbone.load_state_dict(state, strict=False)

    if missing:
        print(f'  [DINOv3 weights] Missing keys ({len(missing)}):')
        for k in missing[:5]:
            print(f'    {k}')
    if unexpected:
        print(f'  [DINOv3 weights] Unexpected keys ({len(unexpected)}):')
        for k in unexpected[:5]:
            print(f'    {k}')

    if cfg.get('lock_backbone', False):
        model.lock_backbone()
    return model, size_key


def _build_optimizer(model, cfg):
    return AdamW(
        [
            {'params': [p for p in model.backbone.parameters()
                        if p.requires_grad],
             'lr': cfg['lr']},
            {'params': [p for n, p in model.named_parameters()
                        if 'backbone' not in n],
             'lr': cfg['lr'] * cfg['lr_multi']},
        ],
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01,
    )


@torch.no_grad()
def _ema_update(model, model_ema, ema_ratio):
    for p, p_ema in zip(model.parameters(), model_ema.parameters()):
        p_ema.copy_(p_ema * ema_ratio + p.detach() * (1 - ema_ratio))
    for b, b_ema in zip(model.buffers(), model_ema.buffers()):
        b_ema.copy_(b_ema * ema_ratio + b.detach() * (1 - ema_ratio))

@torch.no_grad()
def visualize_tsne(model, dataloader, cfg, epoch, save_path):

        model.eval()

        features = []
        labels = []

        max_samples = 10000

        for batch in dataloader:

            img = batch[0]
            mask = batch[1]
            img = torch.nn.functional.interpolate(
                img,
                size=(504, 504),
                mode='bilinear',
                align_corners=False
            )

            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(1).float(),
                size=(504, 504),
                mode='nearest'
            ).squeeze(1).long()
            img = img.cuda()
            mask = mask.cuda()

            pred, feat = model(
                img,
                return_features=True
            )

            B, C, H, W = feat.shape

            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(1).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze(1).long()

            feat = feat.permute(
                0, 2, 3, 1
            ).reshape(-1, C)

            mask = mask.reshape(-1)

            valid = mask != 255

            feat = feat[valid]
            mask = mask[valid]

            if feat.shape[0] > 2000:
                idx = torch.randperm(
                    feat.shape[0],
                    device=feat.device
                )[:2000]

                feat = feat[idx]
                mask = mask[idx]

            features.append(
                feat.cpu()
            )

            labels.append(
                mask.cpu()
            )

            if len(torch.cat(features)) > max_samples:
                break

        features = torch.cat(features)
        labels = torch.cat(labels)

        print(
            "TSNE samples:",
            features.shape
        )

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            init='pca',
            random_state=0
        )

        emb = tsne.fit_transform(
            features.numpy()
        )

        plt.figure(
            figsize=(8, 8)
        )

        num_classes = cfg['nclass']

        for c in range(num_classes):
            idx = labels.numpy() == c

            plt.scatter(
                emb[idx, 0],
                emb[idx, 1],
                s=5,
                label=str(c),
                alpha=0.7
            )

        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(fontsize=8)

        plt.axis('off')

        plt.title(
            f"Learned Representation Epoch {epoch}"
        )

        os.makedirs(
            os.path.join(save_path, "tsne"),
            exist_ok=True
        )

        plt.savefig(
            os.path.join(
                save_path,
                "tsne",
                f"tsne_epoch_{epoch}.png"
            ),
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        model.train()

def main():
    args = parser.parse_args()
    cfg  = yaml.load(open(args.config), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    local_rank = int(os.environ['LOCAL_RANK'])

    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat({**cfg, **vars(args),
                                                   'ngpus': world_size})))
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    use_env_head = args.env_csv is not None
    num_env = 5 if use_env_head else None   # 5 fields: landslide, building, road, water, vegetation
    model, size_key = _build_model(cfg, num_env_classes=num_env)
    optimizer = _build_optimizer(model, cfg)

    if rank == 0:
        logger.info('Total params:   {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M'.format(count_params(model.head)))
        if use_env_head:
            logger.info('Env head params: {:.1f}M (weight={}, csv={})\n'.format(
                count_params(model.env_head), args.env_loss_weight, args.env_csv))
        else:
            logger.info('')

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False,
        output_device=local_rank, find_unused_parameters=True,
    )

    # EMA teacher (same as original FixMatch)
    model_ema = deepcopy(model)
    model_ema.eval()
    for p in model_ema.parameters():
        p.requires_grad = False

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(
            **cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(
            **cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError(cfg['criterion']['name'])

    # Reduction='none' so we can apply per-pixel weights
    criterion_u = nn.CrossEntropyLoss(reduction='none', ignore_index=255).cuda(local_rank)
    criterion_x_pixel = nn.CrossEntropyLoss(
        reduction='none', ignore_index=cfg['criterion']['kwargs'].get('ignore_index', 255)
    ).cuda(local_rank)

    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    use_sam = True

    if sam_version == 'sam3_hf':
        if rank == 0:
            logger.info(f'SAM version: sam3_hf (HF Transformers pipeline)\n')
    elif sam_version == 'sam2':
        use_sam = (sam_checkpoint is not None) and os.path.exists(sam_checkpoint or '')
        if not use_sam and rank == 0:
            logger.warning('SAM2 checkpoint not found → falling back to '
                           'teacher pseudo-labels only.\n')
    elif sam_version == 'sam3_official':
        use_sam = (sam_checkpoint is not None) and os.path.exists(sam_checkpoint or '')
        if not use_sam and rank == 0:
            logger.warning('SAM3 Official checkpoint not found → falling back to '
                           'teacher pseudo-labels only.\n')

    sam_generator: Optional[_BaseSAMPseudoLabelGenerator] = None
    if use_sam:
        # Only instantiate on rank-0 to avoid OOM; other ranks get labels via broadcast
        if rank == 0:
            sam_kwargs = {
                'conf_threshold': cfg.get('conf_thresh', 0.70),
                'purity_threshold': cfg.get('sam_purity_thresh', 0.60),
            }
            if sam_version == 'sam3_hf':
                sam_kwargs['sam3_model_name'] = args.sam_model_name
            elif sam_version == 'sam3_official' and args.sam_text_prompts:
                sam_kwargs['text_prompts'] = args.sam_text_prompts

            sam_generator = create_sam_generator(
                sam_version=sam_version,
                nclass=cfg['nclass'],
                sam_checkpoint=sam_checkpoint if sam_version != 'sam3_hf' else None,
                sam_config=args.sam_config if sam_version == 'sam2' else None,
                device='cuda',
                **sam_kwargs,
            )
        if rank == 0:
            logger.info(f'SAM ({sam_version}) Pseudo-Label Generator enabled.\n')

    trainset_u = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u',
        cfg['crop_size'], args.unlabeled_id_path,
    )
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l',
        cfg['crop_size'], args.labeled_id_path,
        nsample=len(trainset_u.ids),
        env_csv_path=args.env_csv,
    )
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l  = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True,
        num_workers=4, drop_last=True, sampler=trainsampler_l,
    )
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u  = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True,
        num_workers=4, drop_last=True, sampler=trainsampler_u,
    )
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader  = DataLoader(
        valset, batch_size=1, pin_memory=True,
        num_workers=1, drop_last=False, sampler=valsampler,
    )

    valset_labeled = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val',
        id_path=args.labeled_id_path,
    )
    valset_unlabeled = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val',
        id_path=args.unlabeled_id_path,
    )
    valsampler_l = torch.utils.data.distributed.DistributedSampler(valset_labeled)
    valloader_l  = DataLoader(
        valset_labeled, batch_size=1, pin_memory=True,
        num_workers=1, drop_last=False, sampler=valsampler_l,
    )
    valsampler_u = torch.utils.data.distributed.DistributedSampler(valset_unlabeled)
    valloader_u_eval = DataLoader(
        valset_unlabeled, batch_size=1, pin_memory=True,
        num_workers=1, drop_last=False, sampler=valsampler_u,
    )

    total_iters = len(trainloader_u) * cfg['epochs']

    previous_best = previous_best_ema = 0.0
    best_epoch = best_epoch_ema = 0
    epoch = -1

    ckpt_path = os.path.join(args.save_path, 'latest.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        # strict=False so env_head (new) or other added heads won't break resume
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        if rank == 0 and (missing or unexpected):
            logger.info(f'Checkpoint load: missing={missing}, unexpected={unexpected}')
        model_ema.load_state_dict(ckpt['model_ema'], strict=False)
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except Exception as _e:
            if rank == 0:
                logger.info(f'Optimizer state incompatible (new env_head params). '
                            f'Starting fresh optimizer. Error: {_e}')
        epoch            = ckpt['epoch']
        previous_best    = ckpt['previous_best']
        previous_best_ema= ckpt['previous_best_ema']
        best_epoch       = ckpt['best_epoch']
        best_epoch_ema   = ckpt['best_epoch_ema']
        if rank == 0:
            logger.info('Loaded checkpoint @ epoch %d\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info(
                '===========> Epoch: {:}, Prev best: {:.2f} @ep-{:}, '
                'EMA: {:.2f} @ep-{:}'.format(
                    epoch, previous_best, best_epoch,
                    previous_best_ema, best_epoch_ema,
                )
            )

        m_loss      = AverageMeter()
        m_loss_x    = AverageMeter()
        m_loss_u    = AverageMeter()
        m_mask_ratio= AverageMeter()
        m_sam_ratio = AverageMeter()   # fraction of pixels covered by SAM labels
        m_loss_env  = AverageMeter()   # env auxiliary loss
        m_cw_avg    = AverageMeter()   # SAGT: mean consensus weight on unlabeled

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        model.train()

        loader = zip(trainloader_l, trainloader_u)

        for i, (
            (img_x, mask_x, env_x),
            (img_u_w, img_u_s, _, ignore_mask, cutmix_box, _)
        ) in enumerate(loader):

            img_x,  mask_x      = img_x.cuda(),      mask_x.cuda()
            img_u_w, img_u_s    = img_u_w.cuda(),     img_u_s.cuda()
            ignore_mask, cutmix_box = ignore_mask.cuda(), cutmix_box.cuda()
            env_x = env_x.cuda()

            with torch.no_grad():
                # Teacher on weak-augmented unlabelled images
                pred_u_w_logits = model_ema(img_u_w).detach()
                conf_u_w = pred_u_w_logits.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w_logits.argmax(dim=1)              # teacher fallback

                if use_sam and sam_generator is not None:
                    # Generate on rank-0; broadcast to other ranks
                    if rank == 0:
                        sam_labels, sam_quality = sam_generator.generate(
                            img_u_w, model_ema
                        )
                    else:
                        sam_labels  = torch.zeros_like(mask_u_w)
                        sam_quality = torch.zeros_like(mask_u_w, dtype=torch.bool)

                    if world_size > 1:
                        torch.distributed.broadcast(sam_labels,  src=0)
                        torch.distributed.broadcast(sam_quality, src=0)

                    # Unify: use SAM label where quality is high,
                    # else fall back to teacher pseudo-label (if conf OK)
                    final_u_label = torch.where(
                        sam_quality,
                        sam_labels,
                        torch.where(
                            conf_u_w >= cfg['conf_thresh'],
                            mask_u_w,
                            torch.full_like(mask_u_w, 255),
                        )
                    )
                    sam_coverage = (sam_quality.float().sum()
                                    / (ignore_mask != 255).sum().float().clamp(min=1))

                    if args.consensus_weight:
                        teacher_active = conf_u_w >= cfg['conf_thresh']
                        agree = (sam_labels == mask_u_w)
                        pixel_weight = torch.zeros_like(
                            mask_u_w, dtype=torch.float32)
                        pixel_weight[sam_quality & teacher_active & agree] = 1.0
                        pixel_weight[sam_quality & teacher_active & ~agree] = \
                            args.cw_disagree
                        pixel_weight[sam_quality & ~teacher_active] = \
                            args.cw_sam_only
                        pixel_weight[~sam_quality & teacher_active] = \
                            args.cw_teacher_only
                    else:
                        pixel_weight = None  # signal: use original binary mask
                else:
                    # Pure FixMatch teacher pseudo-labels
                    final_u_label = torch.where(
                        conf_u_w >= cfg['conf_thresh'],
                        mask_u_w,
                        torch.full_like(mask_u_w, 255),
                    )
                    sam_coverage = torch.tensor(0.0)
                    pixel_weight = None  # no SAM → no consensus

            img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = \
                img_u_s.flip(0)[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]

            # Apply CutMix to pseudo-labels & confidence maps
            mask_u_cut   = final_u_label.clone()
            conf_u_cut   = conf_u_w.clone()
            ignore_cut   = ignore_mask.clone()

            mask_u_cut[cutmix_box == 1]  = final_u_label.flip(0)[cutmix_box == 1]
            conf_u_cut[cutmix_box == 1]  = conf_u_w.flip(0)[cutmix_box == 1]
            ignore_cut[cutmix_box == 1]  = ignore_mask.flip(0)[cutmix_box == 1]

            if pixel_weight is not None:
                weight_u_cut = pixel_weight.clone()
                weight_u_cut[cutmix_box == 1] = pixel_weight.flip(0)[cutmix_box == 1]
            else:
                weight_u_cut = None

            num_lb  = img_x.shape[0]
            num_ulb = img_u_s.shape[0]

            combined = torch.cat((img_x, img_u_s))            # unified path
            if use_env_head:
                pred_all, env_logits_all = model(combined, return_env=True)
            else:
                pred_all = model(combined)
            pred_x, pred_u_s = pred_all.split([num_lb, num_ulb])

            criterion_pixel = nn.CrossEntropyLoss(
                reduction='none',
                ignore_index=255
            ).cuda(local_rank)

            pred_all = torch.cat([pred_x, pred_u_s], dim=0)

            mask_x_fixed = mask_x.clone()
            mask_x_fixed[mask_x_fixed == cfg['criterion']['kwargs'].get('ignore_index', 255)] = 255

            mask_all = torch.cat([mask_x_fixed, mask_u_cut], dim=0)

            valid_x = mask_x_fixed != 255
            valid_u = (ignore_cut != 255) & (mask_u_cut != 255)

            if weight_u_cut is not None:
                weight_x = valid_x.float()
                weight_u = weight_u_cut * valid_u.float()
                weight_all = torch.cat([weight_x, weight_u], dim=0)
            else:
                valid_all = torch.cat([valid_x, valid_u], dim=0)
                weight_all = valid_all.float()

            loss_pixel = criterion_pixel(pred_all, mask_all)

            loss = (
                       loss_pixel * weight_all
                   ).sum() / weight_all.sum().clamp(min=1)

            loss_env = torch.tensor(0.0, device=loss.device)
            if use_env_head:
                # env_logits_all: (B_total, 5), only first num_lb are labelled
                env_pred = env_logits_all[:num_lb]          # (num_lb, 5)
                env_valid = (env_x >= 0).all(dim=1)          # (num_lb,) mask
                if env_valid.any():
                    criterion_bce = nn.BCEWithLogitsLoss(reduction='none').cuda(local_rank)
                    bce = criterion_bce(env_pred, env_x.clamp(min=0))  # (num_lb, 5)
                    # Only average over valid samples and fields
                    env_valid_f = env_valid.float().unsqueeze(1)  # (num_lb, 1)
                    loss_env = (bce * env_valid_f).sum() / (env_valid_f.sum() * 5).clamp(min=1)
                    loss = loss + args.env_loss_weight * loss_env

            torch.distributed.barrier()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = epoch * len(trainloader_u) + i
            lr    = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * cfg['lr_multi']

            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            _ema_update(model, model_ema, ema_ratio)

            m_loss.update(loss.item())
            m_loss_env.update(loss_env.item() if isinstance(loss_env, torch.Tensor) else loss_env)

            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) &
                          (ignore_mask != 255)).sum().item() / \
                         (ignore_mask != 255).sum().clamp(min=1).item()
            m_mask_ratio.update(mask_ratio)
            m_sam_ratio.update(sam_coverage.item()
                               if isinstance(sam_coverage, torch.Tensor)
                               else sam_coverage)

            # SAGT: track mean consensus weight
            if pixel_weight is not None:
                m_cw_avg.update(pixel_weight[
                    (ignore_mask != 255)].mean().item())
            else:
                m_cw_avg.update(mask_ratio)  # fallback to binary ratio

            if rank == 0:
                writer.add_scalar('train/loss',       loss.item(),     iters)
                writer.add_scalar('train/mask_ratio', mask_ratio,      iters)
                writer.add_scalar('train/sam_coverage', m_sam_ratio.avg, iters)
                writer.add_scalar('train/loss_env',   m_loss_env.avg,  iters)
                writer.add_scalar('train/cw_avg',      m_cw_avg.avg,    iters)

                if i % (len(trainloader_u) // 8) == 0:
                    logger.info(
                        'Iters: {:4d} | LR: {:.2e} | Loss: {:.3f} '
                        ' Mask: {:.3f} | SAM: {:.3f} | CW: {:.3f} | ENV: {:.3f}'.format(
                            i, lr,
                            m_loss.avg,
                            m_mask_ratio.avg, m_sam_ratio.avg,
                            m_cw_avg.avg, m_loss_env.avg,
                        )
                    )
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        _ps = (model.module if hasattr(model, 'module') else model).patch_size
        mIoU, iou_class, mF1, f1_class, kappa = evaluate(model, valloader, eval_mode, cfg, multiplier=_ps)

        if rank == 0 and epoch % 5 == 0:
            visualize_tsne(
                model.module if hasattr(model, 'module') else model,
                valloader,
                cfg,
                epoch,
                args.save_path
            )
        # mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=14)
        mIoU_ema, iou_class_ema, mF1_ema, f1_class_ema, kappa_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=_ps)

        mIoU_lb,  iou_class_lb,  _, _, _  = evaluate(model, valloader_l,      eval_mode, cfg, multiplier=_ps)
        mIoU_ulb, iou_class_ulb, _, _, _  = evaluate(model, valloader_u_eval, eval_mode, cfg, multiplier=_ps)
        mIoU_lb_ema,  iou_class_lb_ema,  _, _, _ = evaluate(model_ema, valloader_l,      eval_mode, cfg, multiplier=_ps)
        mIoU_ulb_ema, iou_class_ulb_ema, _, _, _ = evaluate(model_ema, valloader_u_eval, eval_mode, cfg, multiplier=_ps)

        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info(
                    '***** Evaluation ***** >>>> '
                    'Class [{:} {:}] IoU: {:.2f}, EMA: {:.2f}'.format(
                        cls_idx, CLASSES[cfg['dataset']][cls_idx],
                        iou, iou_class_ema[cls_idx],
                    )
                )
            logger.info(
                '***** Evaluation {} ***** >>>> '
                'MeanIoU: {:.2f}, EMA: {:.2f}\n'.format(eval_mode, mIoU, mIoU_ema)
            )
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, MeanF1: {:.2f}, Kappa: {:.4f}\n'.format(
                eval_mode, mIoU, mF1, kappa))

            # Partition mIoU log
            logger.info(
                '***** Partition mIoU ***** >>>> '
                'Labeled: {:.2f} (EMA {:.2f}) | '
                'Unlabeled: {:.2f} (EMA {:.2f})\n'.format(
                    mIoU_lb, mIoU_lb_ema, mIoU_ulb, mIoU_ulb_ema,
                )
            )
            for cls_idx in range(len(iou_class_lb)):
                logger.info(
                    '  Class [{:} {:}] '
                    'Labeled IoU: {:.2f} (EMA {:.2f}) | '
                    'Unlabeled IoU: {:.2f} (EMA {:.2f})'.format(
                        cls_idx, CLASSES[cfg['dataset']][cls_idx],
                        iou_class_lb[cls_idx],  iou_class_lb_ema[cls_idx],
                        iou_class_ulb[cls_idx], iou_class_ulb_ema[cls_idx],
                    )
                )

            writer.add_scalar('eval/mIoU',     mIoU,     epoch)
            writer.add_scalar('eval/mF1', mF1, epoch)
            writer.add_scalar('eval/kappa', kappa, epoch)
            writer.add_scalar('eval/mIoU_ema', mIoU_ema, epoch)
            # Partition scalars
            writer.add_scalar('eval/mIoU_labeled',      mIoU_lb,      epoch)
            writer.add_scalar('eval/mIoU_labeled_ema',  mIoU_lb_ema,  epoch)
            writer.add_scalar('eval/mIoU_unlabeled',    mIoU_ulb,     epoch)
            writer.add_scalar('eval/mIoU_unlabeled_ema',mIoU_ulb_ema, epoch)
            for ci, iou in enumerate(iou_class):
                writer.add_scalar(f'eval/{CLASSES[cfg["dataset"]][ci]}_IoU',     iou,               epoch)
                writer.add_scalar(f'eval/{CLASSES[cfg["dataset"]][ci]}_IoU_ema', iou_class_ema[ci], epoch)

        is_best = mIoU >= previous_best
        previous_best     = max(mIoU,     previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if mIoU     == previous_best:     best_epoch     = epoch
        if mIoU_ema == previous_best_ema: best_epoch_ema = epoch

        if rank == 0:
            ckpt = {
                'model':                   model.state_dict(),
                'model_ema':               model_ema.state_dict(),
                'optimizer':               optimizer.state_dict(),
                'epoch':                   epoch,
                'previous_best':           previous_best,
                'previous_best_ema':       previous_best_ema,
                'best_epoch':              best_epoch,
                'best_epoch_ema':          best_epoch_ema,
            }
            torch.save(ckpt, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(ckpt, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
