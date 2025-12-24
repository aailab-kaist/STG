import torch
from PIL import Image
import open_clip
import argparse
from tqdm import tqdm
import clip
import numpy as np
import os
import pandas as pd
import torch

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

class text_image_pair(torch.utils.data.Dataset):
    def __init__(self, dir_path, csv_path, resolution=1, aes=False):
        """

        Args:
            dir_path: the path to the stored images
            file_path:
        """
        self.dir_path = dir_path
        self.aes = aes
        df = pd.read_csv(csv_path)
        self.text_description = df['caption']
        if resolution != 1:
            _, _, self.preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k', force_image_size=resolution)
        else:
            _, _, self.preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')

        if self.aes:
            _, self.preprocess2 = clip.load("ViT-L/14", device='cuda')  # RN50x64

    def __len__(self):
        return len(self.text_description)

    def __getitem__(self, idx):

        img_path = os.path.join(self.dir_path, f'{idx}.png')
        raw_image = Image.open(img_path)
        image = self.preprocess(raw_image).squeeze().float()
        if self.aes:
            image2 = self.preprocess2(raw_image).squeeze().float()
        else:
            image2 = image
        text = self.text_description[idx]
        return image, image2, text

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            # nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--text_path', type=str, default='./generated_images/subset.csv')
parser.add_argument('--img_path', type=str, default='./generated_images/subset')
parser.add_argument('--num_eval', type=int, default=5000)
parser.add_argument('--aes_path', type=str, default='./clip-refs/aesthetic-model.pth')
parser.add_argument('--aes', action='store_true', default=False)
args = parser.parse_args()

# define dataset / data_loader
text2img_dataset = text_image_pair(dir_path=args.img_path, csv_path=args.text_path)
text2img_loader = torch.utils.data.DataLoader(dataset=text2img_dataset, batch_size=args.bs, shuffle=False)

print("total length:", len(text2img_dataset))
model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
model = model.cuda().eval()
tokenizer = open_clip.get_tokenizer('ViT-g-14')
if args.aes:
    model2, _ = clip.load("ViT-L/14", device='cuda')  #RN50x64
    model2 = model2.eval()

    model_aes = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    #s = torch.load("./clip-refs/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
    s = torch.load(args.aes_path)

    model_aes.load_state_dict(s)
    model_aes.to("cuda")
    model_aes.eval()

cnt = 0.
total_clip_score = 0.
if args.aes:
    total_aesthetic_score = 0.

scores = np.zeros((args.num_eval,), dtype=np.float32)
with torch.no_grad(), torch.cuda.amp.autocast():
    for idx, (image, image2, text) in tqdm(enumerate(text2img_loader)):
        image = image.cuda().float()
        if args.aes:
            image2 = image2.cuda().float()
        text = list(text)
        text = tokenizer(text).cuda()
        # print('text:')
        # print(text.shape)
        image_features = model.encode_image(image).float()
        text_features = model.encode_text(text).float()
        # (bs, 1024)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        total_clip_score += (image_features * text_features).sum()
        scores[args.bs*idx:args.bs*idx+len(image)] = (image_features * text_features).sum(dim=-1).cpu().numpy()
        if args.aes:
            image_features = model2.encode_image(image)
            im_emb_arr = normalized(image_features.cpu().detach().numpy())
            total_aesthetic_score += model_aes(torch.from_numpy(im_emb_arr).to(image.device).type(torch.cuda.FloatTensor)).sum()

        cnt += len(image)

        if cnt >= args.num_eval:
            print(f'Evaluation complete! NUM EVAL: {cnt}')
            break

np.savez(f'{args.img_path}/clip_scores.npz', scores=scores)

print("Average ClIP score :", total_clip_score.item() / cnt)
if args.aes:
    print("Average Aesthetic score :", total_aesthetic_score.item() / cnt)