# Dropout Injection At Test Time For Post Hoc Uncertainty Quantification In Neural Networks

This repository contains the reproducibility of experiments on UCI datasets conducted in our [Dropout Injection](https://www.arxiv.com "Dropout Injection") paper. 


[![Open DropoutInjection in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MKMxabc2Gwvkl42aLutzVCj2VILblulW#scrollTo=gxVrKIwcNBFH) Colab Coming Soon!

## Methodology - From Deterministic to Probabilistic :magic_wand:
1. Take your already-trained neural network
2. Inject dropout layers with a relatively small dropout rate
3. Scale the corresponding uncertainty measure usign a validation set

Now you turned your deterministic network to a probabilistic one without any further, time-consuming, retraining process!

## How to Reproduce the Experiments :page_with_curl:
You need to install the required library, including `torch`, `numpy`, `matplotlib` and `json`. We will add some further details about the requirements soon.

The experiments have been conducted on 8 UCI datasets:
- Boston Housing              (bostonHousing)
- Concrete                    (concrete)
- Energy                      (energy)
- Kin8nm                      (kin8nm)
- Power Plant                 (power-plant)
- Protein Tertiary Structure  (protein-tertiary-structure)
- Wine Quality Red            (wine-quality-red)
- Yacht                       (yacht)

For reproducing the experiment on a given dataset it is sufficient to run the training and the test steps, by running the `main.py` script.

### Training phase
```
python main.py --dataset 'bostonHousing' --mode 'train'
```

### Test phase
```
python main.py --dataset 'bostonHousing' --mode 'test'
```