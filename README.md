# A Comparative Analysis of Computational Drug Repurposing Approaches -- Proposing a Novel Tensor-Matrix-Tensor Factorization Method
_by A. Zabihian, J. Asghari, M. Hooshmand, S. Gharaghani_
![TMTformuation](https://github.com/BioinformaticsIASBS/Tensor/assets/44480584/95ac4d24-cbd2-4a58-814b-8138546de82e)
Gain access to the preprint version of the manuscript [here](https://www.researchsquare.com/article/rs-3816066/latest).

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
    * Two optimizers each developed based on its input data,
    * A module to help assess model performance. 

`TMTF.py` is essentially a pipeline that utilizes the data and the modules to find new DTIs.


## Execuation
Get a full description of the arguments and options involved in the pipeline:
```
TMTF.py -h
```
### Examples
Choose a dataset:
```
TMTF.py GPCR
```

Specify the model formulation (only applicable to DTINet) and the number of latent variables:
```
TMTF.py DTINET --form_no 3 --f_size 32
```

Set the stopping criterion and learning rate:
```
TMTF.py E --epsilon 1e-5 --alpha 0.1
```

Log the optimization process:
```
TMTF.py NR --log
```

The output of `TMTF.py` is the probabilistic predictions matrix $\hat{X}$ saved in the same directory as the file. Keep in mind that the TMT model predicts non-thresholded decision values. So to enable the computation of some of the metrics, at some point, these values should be converted into binary labels. To do this, you may take advantage of `Utilities/classifier_evaluator.py`.
Have a look at its different settings in detail:
```
classifier_evaluator.py -h
```
### Examples
Specify the path of $y$, e.g., `Data/Gold standard dataset/nr_admat_dgc.txt`, and $\hat{y}$:
```
classifier_evaluator.py "../Data/Gold standard dataset/nr_admat_dgc.txt" "../X_hat.txt"
```

Change the method of threshold tuning and see the optimal decision threshold:
```
classifier_evaluator.py "../Data/Gold standard dataset/nr_admat_dgc.txt" "../X_hat.txt" --action tht --tht_mode roc
```

Plot ROC/PR curves:
```
classifier_evaluator.py "../Data/Gold standard dataset/nr_admat_dgc.txt" "../X_hat.txt" --action pr
```

You may find the computed metrics in the same path as the `classifier_evaluator.py` file.




## Citation
```
```