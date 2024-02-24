from diffusers import StableDiffusionPipeline
import torch
from typing import Optional, Union, Dict, Any, List, Type
import components
from pytorch_lightning import seed_everything
import logging
from rich.logging import RichHandler
from rich.progress import track
from itertools import chain
import fire
import numpy as np
from dataset import load_member_data
from collections import defaultdict
from torchmetrics.classification import BinaryAUROC, BinaryROC


class EpsGetter(components.EpsGetter):
    def __call__(self, xt: torch.Tensor, condition: torch.Tensor = None, noise_level=None, t: int = None) -> torch.Tensor:
        # t = torch.ones([xt.shape[0]], device=xt.device).long() * t
        return self.model(t, condition, latents=xt)


class MyStableDiffusionPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def prepare_latent(self, img):
        latents = self.vae.encode(img).latent_dist.sample()
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def encode_input_prompt(self, prompt, do_classifier_free_guidance=True):
        text_encoder_lora_scale = None
        prompt_embeds = self._encode_prompt(
            prompt,
            'cuda',
            1,
            do_classifier_free_guidance,
            None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=text_encoder_lora_scale,
        )
        return prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        t,
        prompt_embeds,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        do_classifier_free_guidance = guidance_scale > 1.0

        # 7. Denoising loop
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred

    def get_image(self, latents):
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)
        return image


def get_FLAGS():

    def FLAGS(x): return x
    FLAGS.T = 1000
    FLAGS.ch = 128
    FLAGS.ch_mult = [1, 2, 2, 2]
    FLAGS.attn = [1]
    FLAGS.num_res_blocks = 2
    FLAGS.dropout = 0.1
    FLAGS.beta_1 = 0.00085
    FLAGS.beta_T = 0.012

    return FLAGS


attackers: Dict[str, Type[components.DDIMAttacker]] = {
    "SecMI": components.SecMIAttacker,
    "PIA": components.PIA,
    "naive": components.NaiveAttacker,
    "PIAN": components.PIAN,
}

DEVICE = 'cuda'


@torch.no_grad()
def main(attacker_name,
         dataset,
         checkpoint,
         norm_p,
         attack_num=50, interval=10,
         save_logger=None,
         seed=0):
    seed_everything(seed)

    FLAGS = get_FLAGS()

    logger = logging.getLogger()
    logger.disabled = True if save_logger else False
    logger.setLevel(logging.INFO)
    logger.addHandler(RichHandler())

    logger.info("loading model...")
    model = MyStableDiffusionPipeline.from_pretrained(checkpoint, torch_dtype=torch.float32)
    model = model.to(DEVICE)
    # model.eval()

    def attacker_wrapper(attack):
        def wrapper(x, condition=None):
            x = model.prepare_latent(x)
            if 'none' in dataset:
                condition = ['none'] * len(condition)
            if condition is not None:
                condition = model.encode_input_prompt(condition)
            return attack(x, condition)

        return wrapper

    logger.info("loading dataset...")
    _, _, train_loader, test_loader = load_member_data(dataset_name=dataset, batch_size=4)

    attacker = attackers[attacker_name](
        torch.from_numpy(np.linspace(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T)).to(DEVICE), interval, attack_num, EpsGetter(model), lambda x: x * 2 - 1)
    attacker = attacker_wrapper(attacker)

    logger.info("attack start...")
    members, nonmembers = defaultdict(list), defaultdict(list)

    for member, nonmember in track(zip(train_loader, chain(*([test_loader]))), total=len(test_loader)):
        member_condition, nonmenmer_condition = member[1], nonmember[1]
        member, nonmember = member[0].to(DEVICE), nonmember[0].to(DEVICE)

        intermediate_reverse_member, intermediate_denoise_member = attacker(member, member_condition)
        intermediate_reverse_nonmember, intermediate_denoise_nonmember = attacker(nonmember, nonmenmer_condition)

        members[norm_p].append(((intermediate_reverse_member - intermediate_denoise_member).abs() ** norm_p).flatten(2).sum(dim=-1))
        nonmembers[norm_p].append(((intermediate_reverse_nonmember - intermediate_denoise_nonmember).abs() ** norm_p).flatten(2).sum(dim=-1))

        members[norm_p] = [torch.cat(members[norm_p], dim=-1)]
        nonmembers[norm_p] = [torch.cat(nonmembers[norm_p], dim=-1)]

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
