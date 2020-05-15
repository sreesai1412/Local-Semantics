# GAN-Local-Semantics
### Data
```
mkdir data && cd data
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz && tar -xf CUB_200_2011.tgz
```

### Pretrained VGG16 weights
You may create a directory `vgg16` and download the pretriained weights there.

### To run
```
python feature_clustering.py -eid sph_k_means -k 15
```
