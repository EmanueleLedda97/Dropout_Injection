# Dropout Injection At Test Time For Post Hoc Uncertainty Quantification In Neural Networks

This repository contains the reproducibility of experiments on UCI datasets conducted in our [Dropout Injection](https://www.arxiv.com "Dropout Injection") paper. 


[![Open DropoutInjection in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MKMxabc2Gwvkl42aLutzVCj2VILblulW#scrollTo=gxVrKIwcNBFH)

## Methodology - From Deterministic to Probabilistic :magic_wand:
1. Take your already-trained neural network
2. Inject dropout layers with a relatively small dropout rate
3. Scale the corresponding uncertainty measure usign a validation set

Now you turned your deterministic network to a probabilistic one!
