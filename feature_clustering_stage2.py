# -*- coding: utf-8 -*-
import torch
import numpy as np
from factor_catalog import FactorCatalog
from glob import glob
from PIL import Image as im
from torchvision import transforms
import torchvision.models
import random, time
import matplotlib.pyplot as plt
from spherical_gmm import GMMWrapper
import argparse
import os
from utils import preprocess_image

SEED = 0
k = 10 # num of clusters
DATASET_DIRECTORY = 'data/CUB_200_2011/images/'
MASK_DIRECTORY = 'data/CUB_200_2011/segmentations/'

VGG_PATH = 'vgg16/vgg16_bn-6c64b313.pth'
BATCH_SIZE = 200
RESULT_DIR = 'results'
parser = argparse.ArgumentParser()
parser.add_argument('-eid', '--experiment-id', type=str)
parser.add_argument('-dd', '--dataset-directory', type=str, default=DATASET_DIRECTORY)
parser.add_argument('-b', '--batch-size', type=int, default=BATCH_SIZE)
parser.add_argument('-s', '--seed', type=int, default=SEED)
parser.add_argument('-k', '--clusters', type=int, default=k)

args = parser.parse_args()

random.seed(args.seed)

directory_output = os.path.join(RESULT_DIR, args.experiment_id)
os.makedirs(directory_output, exist_ok=True)

#
size_tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])

def getImages():
    def _getIm(img_path):
        img = im.open(img_path)
        path = img_path.split('/')
        mask_path = '.'.join('/'.join(path[:2] + ['segmentations'] + path[3:]).split('.')[:-1] + ['png'])
        mask = im.open(mask_path)
        if len(img.getbands()) != 3:
            return None, None
        img = tfm(img)
        mask = size_tfm(mask)
        return img, (np.asarray(mask)/255. > 0.5).astype('uint8')

    img_paths = glob(args.dataset_directory+'/*/*.jpg')
    random.shuffle(img_paths)
    imgs = []
    masks = []
    for img_path in img_paths:
        img, mask = _getIm(img_path) 
        if not isinstance(img, type(None)):
            imgs.append(img)
            masks.append(mask)
        if len(imgs) == args.batch_size+8:
            break
    imgs = torch.stack(imgs).float()
    masks = np.stack(masks, axis=0)
    return imgs, masks


class VGGFeatures(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGFeatures, self).__init__()
        blocks = []

        model = torchvision.models.vgg16_bn()
        state_dict = torch.load(VGG_PATH)
        model.load_state_dict(state_dict)
        blocks = []
        blocks.append(model.features[:34].eval())
        blocks.append(model.features[34:37].eval())
        blocks.append(model.features[37:40].eval())
        blocks.append(model.features[40:43].eval())
        

        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, normalize=False, block_id=2):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)

        if normalize:
            input = (input-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            #target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        x = input
        #feats = [x]
        if block_id < 0:
            return x
        for i, block in enumerate(self.blocks):
            #feats.append(block(feats[-1]))
            x = block(x)
            if i == block_id:
                return x
        return x

fe = VGGFeatures().cuda()
block_id = 3
start = time.time()
imgs, masks = getImages()
imgs_val, masks_val = imgs[args.batch_size:], masks[args.batch_size:]
end = time.time()
print('Loaded %d images (%.3fs)'%(args.batch_size, end-start))
x = imgs.cuda()

idx = 0

postpro = transforms.Compose([transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                std=[1/0.229, 1/0.224, 1/0.225]), transforms.ToPILImage()])

fc = FactorCatalog(args.clusters)
fc2 = FactorCatalog(args.clusters*2+2)

start = time.time()
feats = fe(x, block_id=block_id).detach().cpu()
hms = fc.fit_predict(feats[:args.batch_size], True)

feats2 = fe(x, block_id=block_id).detach().cpu()
hms_onehot = hms.get(feats2.shape[-1])
# THis has to be 0/1, decided by seeing the output of stage 1
mask = torch.argmax(hms_onehot, dim=1, keepdim=True) == 1

hms = fc2.fit_predict(feats2[:args.batch_size], True, mask)
hms_val = fc.predict(feats[args.batch_size:], True)
#hms_val_onehot = hms_val.get(feats2.shape[-1])
#mask_val = torch.argmax(hms_val_onehot, dim=1, keepdim=True) == 1
#hms_val = fc2.predict(feats2[args.batch_size:], True, mask_val)

hms_cold= hms_val.get(224).argmax(dim=1)

idx = 0

# PLOT OUTPUTS
#palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(args.clusters*2+2)])[:, None]
colors = (colors * 255 / (args.clusters*2+1)).numpy().astype("uint8")

nrows, ncols = 4, 4
f, axarr = plt.subplots(nrows, ncols, figsize=(12,12))
for row in axarr:
    for i, col in enumerate(row):
        r = im.fromarray(hms_cold[idx].byte().cpu().numpy())#.resize((224, 224))
        r.putpalette(colors)
        r = np.asarray(r)
        original_img = postpro(imgs_val[idx])
        if i % 2 == 0:
            col.imshow(original_img)
        else:
            original_img = original_img.convert('L')
            col.imshow(original_img, cmap='gray', vmin=0, vmax=255)
            col.imshow(np.asarray(r).astype('uint8'), cmap='viridis', alpha=0.5)
            idx += 1
        col.axis('off')


f.savefig(os.path.join(directory_output, 'sph_k_means_5-3_5-3.png'))
end = time.time()
print('Block Id %d done (%.3fs)'%(block_id+1, end-start))

#del feats
#f_img.savefig(os.path.join(directory_output, 'img_v_blockId.png'))