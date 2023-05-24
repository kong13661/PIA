import numpy as np
import os

import torch
from rich.progress import track
import fire
import logging
from rich.logging import RichHandler
from pytorch_lightning import seed_everything
from typing import Type, Dict
from grad_tts.model import GradTTS
from itertools import chain
import importlib
from grad_tts.text.symbols import symbols
from grad_tts.data import TextMelSpeakerDataset, TextMelSpeakerBatchCollate
from torch.utils.data import DataLoader
from grad_tts.model.utils import fix_len_compatibility
from grad_tts.data import TextMelDataset, TextMelBatchCollate
from torchmetrics.classification import BinaryAUROC, BinaryROC


params_dict = {
    'ljspeech': lambda: importlib.import_module('grad_tts.params_ljspeech'),
    'libritts': lambda: importlib.import_module('grad_tts.params_libritts'),
}


DEVICE = 'cuda'


@torch.no_grad()
def main(checkpoint,
         dataset,
         attacker_name="PIA",
         attack_num=99, interval=10,
         seed=0,
         batch_size=20):
    T = 1000
    seed_everything(seed)

    params = params_dict[dataset]()
    train_filelist_path = params.train_filelist_path
    valid_filelist_path = params.valid_filelist_path
    cmudict_path = params.cmudict_path
    add_blank = params.add_blank

    nsymbols = len(symbols) + 1 if add_blank else len(symbols)
    n_enc_channels = params.n_enc_channels
    filter_channels = params.filter_channels
    filter_channels_dp = params.filter_channels_dp
    n_enc_layers = params.n_enc_layers
    enc_kernel = params.enc_kernel
    enc_dropout = params.enc_dropout
    spk_emb_dim = params.spk_emb_dim
    n_heads = params.n_heads
    window_size = params.window_size

    n_feats = params.n_feats
    n_fft = params.n_fft
    sample_rate = params.sample_rate
    hop_length = params.hop_length
    win_length = params.win_length
    f_min = params.f_min
    f_max = params.f_max
    n_spks = params.n_spks

    dec_dim = params.dec_dim
    beta_min = params.beta_min
    beta_max = params.beta_max
    pe_scale = params.pe_scale
    output_size = fix_len_compatibility(2 * 22050 // 256)
    # output_size = None
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(RichHandler())

    logger.info("initializing model...")
    model = GradTTS(nsymbols, n_spks, None if n_spks == 1 else spk_emb_dim, n_enc_channels,
                    filter_channels, filter_channels_dp,
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                    n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()
    logger.info("loading checkpoint...")
    if 'libritts' in dataset:
        model.load_state_dict(torch.load(checkpoint, map_location=lambda loc, storage: loc)['ckpt'])
    else:
        model.load_state_dict(torch.load(checkpoint, map_location=lambda loc, storage: loc))
    model.eval()

    logger.info("loading dataset...")
    if n_spks > 1:
        train_dataset = TextMelSpeakerDataset(train_filelist_path, cmudict_path, add_blank,
                                              n_fft, n_feats, sample_rate, hop_length,
                                              win_length, f_min, f_max)
        batch_collate = TextMelSpeakerBatchCollate()
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  collate_fn=batch_collate, drop_last=True,
                                  shuffle=False)

        test_dataset = TextMelSpeakerDataset(valid_filelist_path, cmudict_path, add_blank,
                                             n_fft, n_feats, sample_rate, hop_length,
                                             win_length, f_min, f_max)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 collate_fn=batch_collate, drop_last=True,
                                 shuffle=False)
    else:
        train_dataset = TextMelDataset(train_filelist_path, cmudict_path, add_blank,
                                       n_fft, n_feats, sample_rate, hop_length,
                                       win_length, f_min, f_max)
        batch_collate = TextMelBatchCollate()
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  collate_fn=batch_collate, drop_last=True,
                                  shuffle=False)
        test_dataset = TextMelDataset(valid_filelist_path, cmudict_path, add_blank,
                                      n_fft, n_feats, sample_rate, hop_length,
                                      win_length, f_min, f_max)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 collate_fn=batch_collate, drop_last=True,
                                 shuffle=False)

    def recon_score(n_timesteps, terminal_time, batch):
        x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
        y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
        spk = None if 'spk' not in batch else batch['spk'].to(torch.long).cuda()

        attacker = {
            'SecMI': model.decoder.SecMI,
            'naive': model.decoder.naive_attack,
            'PIA': model.decoder.PIA,
            "PIAN": model.decoder.PIAN,
        }

        y_mask, mu_y, y, _ = model.forward_decoder_ahead(x, x_lengths, y, y_lengths, out_size=output_size, spk=spk)
        if hasattr(model, "spk_emb"):
            spk = model.spk_emb(spk)
        return attacker[attacker_name](y, mu_y, y_mask, n_timesteps, 0, 0, terminal_time=terminal_time, spk=spk)

    logger.info("attack start...")
    members, nonmembers = [], []
    count = 0
    for member, nonmember in track(zip(train_loader, chain(*([test_loader]))), total=len(test_loader)):
        members.append(recon_score(attack_num, interval / T * attack_num, member))
        nonmembers.append(recon_score(attack_num, interval / T * attack_num, nonmember))

        members = [torch.cat(members, dim=-1)]
        nonmembers = [torch.cat(nonmembers, dim=-1)]
        count += 1
        if count == 2:
            break

    member = members[0]
    nonmember = nonmembers[0]

    auroc = [BinaryAUROC().cuda()(torch.cat([member[i] / max([member[i].max().item(), nonmember[i].max().item()]), nonmember[i] / max([member[i].max().item(), nonmember[i].max().item()])]), torch.cat([torch.zeros(member.shape[1]).long(), torch.ones(nonmember.shape[1]).long()]).cuda()).item() for i in range(member.shape[0])]
    tpr_fpr = [BinaryROC().cuda()(torch.cat([1 - nonmember[i] / max([member[i].max().item(), nonmember[i].max().item()]), 1 - member[i] / max([member[i].max().item(), nonmember[i].max().item()])]), torch.cat([torch.zeros(member.shape[1]).long(), torch.ones(nonmember.shape[1]).long()]).cuda()) for i in range(member.shape[0])]
    tpr_fpr_1 = [i[1][(i[0] < 0.01).sum() - 1].item() for i in tpr_fpr]
    cp_auroc = auroc[:]
    cp_auroc.sort(reverse=True)
    cp_tpr_fpr_1 = tpr_fpr_1[:]
    cp_tpr_fpr_1.sort(reverse=True)
    print('auc', auroc)
    print('tpr @ 1% fpr', cp_tpr_fpr_1)


if __name__ == '__main__':
    fire.Fire(main)
