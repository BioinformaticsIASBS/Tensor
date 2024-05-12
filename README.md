# A Comparative Analysis of Computational Drug Repurposing Approaches -- Proposing a Novel Tensor-Matrix-Tensor Factorization Method
_by A. Zabihian, J. Asghari, M. Hooshmand, S. Gharaghani_
![TMT formulation](https://github.com/BioinformaticsIASBS/Tensor/assets/44480584/95ac4d24-cbd2-4a58-814b-8138546de82e)
Gain access to the manuscript [here](https://link.springer.com/article/10.1007/s11030-024-10851-7).

## Overview
This repository provides an implementation of the TMT factorization method. It is organized as follows:
* `Data/` contains:
    * `DTINet dataset/`:
        * Association, interaction, and similarity matrices included in the DTINet dataset,
        * Converted similarity matrices based on the existing associations.
    * `Gold standard dataset/`:
        * Interaction and similarity matrices concerning four pharmaceutically useful drug–target classes: enzymes, ion channels, GPCRs, and nuclear receptors.
* `Utilities/` contains:
    * A module that can compute similarity matrices,
    * Two optimizers, each developed based on its input data,
    * Two modules to compute embeddings and positive-to-negative samplings of the data,
    * A module to help assess model performance.

`TMTF.py` and `ML.py` are essentially pipelines that utilize the data and the modules to find new DTIs.

You can also find the implementations of the IEDTI and DEDTI models in this [repository](https://github.com/BioinformaticsIASBS/IEDTI-DEDTI).

## Requirements
```
pip3 install -r requirements.txt
```

## TMTF Pipeline Usage
Get the full description of the arguments and options involved in the pipeline:
```
python3 TMTF.py -h
```

### Examples
Choose a dataset:
```
python3 TMTF.py GPCR
```

Specify the model formulation (only applicable to DTINet) and the number of latent variables:
```
python3 TMTF.py DTINET --form_no 3 --f_size 32
```

Set the stopping criterion and learning rate:
```
python3 TMTF.py E --epsilon 1e-5 --alpha 0.1
```

Log the optimization process:
```
python3 TMTF.py NR --log
```

The output of `TMTF.py` is the probabilistic predictions matrix $\hat{X}$ saved in `Results/TMTF/`. Keep in mind that the TMT model predicts non-thresholded decision values. So to enable the computation of some of the metrics, at some point, these values should be converted into binary labels. To do this, you may take advantage of `Utilities/classifier_evaluator.py`.

## ML Pipeline Usage
See the full list of options:
```
python3 ML.py -h
```

### Examples
Choose a dataset and a ratio of negative sampling:
```
python3 ML.py DTINET --ratio 5
```

Configure the pipeline to compute embeddings and ratio the data (necessary for first-time usage):
```
python3 ML.py DTINET --ratio 5 --comp_embs --comp_ratioed_data
```

Select an algorithm and specify its parameters:
```
python3 ML.py DTINET --ratio 5 --algorithm SVM --kernel rbf --C 10
```

With embeddings calculated and ratioed, train and test a model in its default configuration and compute its average score through all folds:
```
python3 ML.py NR --ratio all --num_folds 5 --algorithm RF --oa_eval 
```

`ML.py` for each fold saves $y$, $\hat{y}$, and probabilistic $\hat{y}$ in `Results/ML/`. You can use the evaluator module to compute the score of each fold. As mentioned above, use `--oa_eval` for an overall evaluation of the model across all folds.

## Model Evaluator Usage
Have a look at its different settings in detail:
```
python3 classifier_evaluator.py -h
```

### Examples
Specify the path of $y$, $\hat{y}$, and probabilistic $\hat{y}$:
```
python3 classifier_evaluator.py \
 "../Results/ML/RF_NR_1-to-1_fold1_y.txt" \
 "../Results/ML/RF_NR_1-to-1_fold1_y-hat.txt" \
 "../Results/ML/RF_NR_1-to-1_fold1_probabilistic-y-hat.txt"
```

In case the model only predicts a probabilistic $\hat{y}$, e.g., TMTF, specify $\hat{y}$ path as `None`:
```
python3 classifier_evaluator.py \
 "../Data/Gold standard dataset/nr_admat_dgc.txt" \
 None \
 "../Results/TMTF/X_hat.txt"
```

Plot ROC/PR curves:
```
python3 classifier_evaluator.py \
 "../Data/Gold standard dataset/nr_admat_dgc.txt" \
 None \
 "../Results/TMTF/X_hat.txt" \
 --action pr
```

You may find the computed scores in the same path as the probabilistic $\hat{y}$.


## Citation
```
@article{
zabihian2024comparative,
title={A comparative analysis of computational drug repurposing approaches: proposing a novel tensor-matrix-tensor factorization method},
author={Zabihian, Arash and Asghari, Javad and Hooshmand, Mohsen and Gharaghani, Sajjad},
journal={Molecular Diversity},
pages={1--20},
year={2024},
publisher={Springer}
}
```