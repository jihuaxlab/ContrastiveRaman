# Contrastive learning for Raman spectroscopy

Code of the contrastive learning framework for Raman spectroscopy, proposed in our [paper](https://doi.org/10.1016/j.chemolab.2025.105384). 

## Requirements

The code has been tested running under Python 3.9.12, with the following packages and their dependencies installed:

```
numpy==1.16.5
pandas==1.4.2
pytorch==1.7.1
sklearn==0.21.3
```

## Usage

```bash
python main.py --c 2 --model CLR
```

Parameter `model` can be one of these models: CNN, LSTM, CLR.

## Datasets

As the dataset in this paper cannot be publicly available, in this repository, we evaluate our model on pathogen Raman spectra datasets in [Yu et al. (2021)](https://doi.org/10.1021/acs.analchem.1c00431), including a binary classification dataset `bin.csv`, and an 8-class classification dataset `multi.csv`.

## Options

We adopt an argument parser by package  `argparse` in Python, and the options for running code are defined as follow:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=300,
                    help='Dimension of representations')
parser.add_argument('--c', type=int, default=2,
                    help='Num of classes')
parser.add_argument('--d', type=int, default=1200,
                    help='Num of spectra dimension')
parser.add_argument('--model', type=str, default='CLR',
                    help='Model')                    

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
```

## Citation

```
@article{shi2025tcraman,
    title = {Transfer contrastive learning for Raman spectra data of urine: Detection of glucose, protein, and prediction of kidney disorders},
    journal = {Chemometr. Intel. Lab. Sys.},
    volume = {261},
    pages = {105384},
    year = {2025}
}
```
