1. Warmup is needed: start from a small η, increase η in a few epochs.
2. Pretrained model is used for ImageNet. (https://pytorch.org/docs/stable/torchvision/models.html)
3. First train the model with ReLU(), then use Clamp, then replace pooling layer with trainable layer, then add quantization layers.
4. Data augmentation is needed for both MNIST and CIFAR-10.
