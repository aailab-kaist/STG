import os
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import ImageReward as RM

parser = argparse.ArgumentParser(description='ImageReward Evaluation')
parser.add_argument('--text_path', type=str, default='subset.csv')
parser.add_argument('--img_path', type=str, default='gen_images')
parser.add_argument('--num_eval', type=int, default=5000, help='number of maximum geneated samples')
parser.add_argument('--bs', type=int, default=4, help='batch size')
args = parser.parse_args()

if __name__ == "__main__":
    df = pd.read_csv(args.text_path)
    all_text = list(df['caption'])
    all_text = all_text[: args.num_eval]

    img_prefix = args.img_path
    generations = [f"{pic_id}.png" for pic_id in range(0, args.num_eval)]
    img_list = [os.path.join(img_prefix, img) for img in generations]

    model = RM.load("ImageReward-v1.0", device='cuda')
    avg_score = 0.

    scores = np.zeros((len(img_list),), dtype=np.float32)

    with torch.no_grad():
        for index in tqdm(range(0, len(img_list), args.bs)):
            score = model.score_batch(all_text[index:index+args.bs], img_list[index:index+args.bs])
            scores[index:index+args.bs] = score.cpu().numpy()
            avg_score += score.sum()
    np.savez(f'{img_prefix}/image_reward_scores.npz', scores=scores)
    avg_score /= len(img_list)
    print(avg_score)
