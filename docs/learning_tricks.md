1. Pretrained model is used for ImageNet. (https://pytorch.org/docs/stable/torchvision/models.html)
2. First train the model with ReLU(), then use Clamp, then add quantization layers.
3. Data augmentation is needed for both MNIST and CIFAR-10. 
4. To get a good model, you may first use data augmentation (set k as 20-50) and dropout rate (0.2-0.3) to train a good initialization. Then, retrain the model with data augmentation (set k as 100-200) and dropout rate (0.1-0.2).

