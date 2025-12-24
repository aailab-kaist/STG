import argparse
import gc

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from pipelines.pipeline_stable_diffusion_stg import StableDiffusionSTGPipeline
import pandas as pd
import numpy as np
import os
import tqdm
from nudenet import NudeDetector_YOLO as SM

device = 'cuda'

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--unlearn_model_path",
        type=str,
        default="checkpoints/DUO",
        required=False,
        help="Path to unlearned lora path",
    )
    parser.add_argument(
        "--exp_type",
        type=str,
        default="base",  # for duo, violence or nudity
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument('--steps', type=int, default=50, help='number of inference steps during sampling')
    parser.add_argument('--generate_seed', type=int, default=6)
    parser.add_argument('--w', type=float, default=7.5)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--max_cnt', type=int, default=5000, help='number of maximum geneated samples')
    parser.add_argument('--save_path', type=str, default='./generated_images')
    parser.add_argument('--scheduler', type=str, default='DDPM')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--text_path', type=str, default='subset.csv')
    parser.add_argument('--adj', type=int, default=0)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--ver_sd', type=str, default='1.5')

    parser.add_argument('--lr_upt_prompt', type=float, default=0.1)
    parser.add_argument('--update_freq', type=int, default=1)
    parser.add_argument('--weight_prior', type=float, default=0.)
    parser.add_argument('--update_itrs', type=str, default='')
    parser.add_argument('--update_intervals', type=str, default='')
    parser.add_argument('--init_org', action='store_true', default=False)
    parser.add_argument('--unsafe_conf', type=float, default=0.)
    parser.add_argument('--nudenet_path', type=str, default='checkpoints/nudenet/nudenet_640m.pt')

    args = parser.parse_args()
    return args


def load_sd_dpo(args):
    if args.ver_sd == '1.5':
        pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    elif args.ver_sd == '1.4':
        pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    else:
        raise NotImplementedError(f"No defined pretrained model name {args.ver_sd}")
    if args.fp16:
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    pipe = StableDiffusionSTGPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=weight_dtype).to(device)

    if args.exp_type == 'violence':
        config_list = ["Blood", "Gun", "Horror", "Suffer"]
        for config_name in config_list:
            lora_path = f'{args.unlearn_model_path}/{config_name}/checkpoint-500/pytorch_lora_weights.safetensors'
            pipe.load_lora_weights(lora_path, adapter_name=config_name)
        pipe.set_adapters(config_list, adapter_weights=[1, 1, 1, 1])
    elif args.exp_type == 'nudity':
        lora_path = f'{args.unlearn_model_path}/Nudity/pytorch_lora_weights.safetensors'
        pipe.load_lora_weights(lora_path)
    else:
        print("No LoRA, base model.")
        pass
    return pipe


if __name__ == '__main__':
    args = parse_args()

    df = pd.read_csv(args.text_path, encoding='cp949')
    all_text = list(df['caption'])
    all_text = all_text[: args.max_cnt]

    num_batches = (len(all_text) - 1) // (args.bs) + 1
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index

    pipe = load_sd_dpo(args)

    if args.fp16:
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    nude_net_model = SM(args.nudenet_path, device='cuda', conf=args.unsafe_conf)

    if args.scheduler == 'DDPM':
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == 'DDIM':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == 'PNDM':
        assert pipe.scheduler._class_name == "PNDMScheduler"
    else:
        raise NotImplementedError
    pipe.safety_checker = None
    pipe = pipe.to(device, weight_dtype)

    generator = torch.Generator(device="cuda").manual_seed(args.generate_seed)
    save_dir = os.path.join(args.save_path,
                            f'scheduler_{args.scheduler}_steps_{args.steps}_w_{args.w}_seed_{args.generate_seed}_name_{args.name}')
    if (args.update_itrs != '') & (args.update_intervals != ''):
        raise NotImplementedError("Please use only one of 'update_itrs' or 'update_intervals'.")
    else:
        if args.update_itrs != '':
            update_itrs = list(map(int, args.update_itrs.split('-')))
        elif args.update_intervals != '':
            update_itrs = list(range(*map(int, args.update_intervals.split('-'))))
        else:
            update_itrs = None

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for cnt, mini_batch in enumerate(tqdm.tqdm(rank_batches, unit='batch')):
        text = list(mini_batch)

        if rank_batches_index[cnt][-1] < args.adj:
            skip_images = True
        else:
            skip_images = False

        outputs = pipe(text, generator=generator, num_inference_steps=args.steps, guidance_scale=args.w,
                     lr_upt_prompt=args.lr_upt_prompt, update_freq=args.update_freq, weight_prior=args.weight_prior,
                     skip_images=skip_images,
                     update_itrs=update_itrs, init_org=args.init_org, nude_net_model=nude_net_model,
        )

        if not skip_images:
            image = outputs.images
            for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
                image[text_idx].save(os.path.join(save_dir, f'{global_idx}.png'))
        gc.collect()
        torch.cuda.empty_cache()
    del pipe
    torch.cuda.empty_cache()
    gc.collect()

d = {'caption': all_text}
df = pd.DataFrame(data=d)
df.to_csv(os.path.join(save_dir, 'subset.csv'))