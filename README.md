# Tensorflow attempt to reazlie ADVENT training for test dataset
### ADVENT is Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation 
Original paper is https://arxiv.org/abs/1811.12833
Main idea is to train a segmentation model with synthetic (source) data, but regularize it with loss of discriminator model, that will use entropy (uncertainty of models prediction) to help segmentation model do better with real (target) data.

### Examples of models prediction

### Unet-like model trained without adverasrial training 
![](readme_images/without_advent/subplot1.jpg?raw=true)
![](readme_images/without_advent/subplot2.jpg?raw=true)
![](readme_images/without_advent/subplot3.jpg?raw=true)
![](readme_images/without_advent/subplot4.jpg?raw=true)

### Unet-like model trained with adverasrial training 
![](readme_images/with_advent/subplot1.jpg?raw=true)
![](readme_images/with_advent/subplot2.jpg?raw=true)
![](readme_images/with_advent/subplot3.jpg?raw=true)
![](readme_images/with_advent/subplot4.jpg?raw=true)