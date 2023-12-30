# A Comparative Analysis of Computational Drug Repurposing Approaches -- Proposing a Novel Tensor-Matrix-Tensor Factorization Method
by A. Zabihian, J. Asghari, M. Hooshmand, S. Gharaghani
![TMTformuation](https://github.com/BioinformaticsIASBS/Tensor/assets/44480584/95ac4d24-cbd2-4a58-814b-8138546de82e)

## Overview
This repository provides an implementation of the TMT factorization method. It is organized as follows:
* `Data/` contains:
    * `DTINet dataset/`:
        * Association, interaction, and similarity matrices included in the DTINet dataset,
        * Converted similarity matrices based on the existing associations,
    * `Gold standard dataset/`:
        * Interaction and similarity matrices concerning four pharmaceutically useful drug–target classes: enzymes, ion channels, GPCRs and nuclear receptors,
* `Utilities/` contains:
    * A module to compute similarity matrices.
    * Two optimizers each developed regarding its input data.
`TMTF.py` is essentially a pipeline that utilizes the data and the modules to find new DTIs.


## Execuation
Get a full description of the arguments and options involved in the pipeline:
```TMTF.py -help```

### Examples
Default configuration:
```TMTF.py GPCR```

Specify the model formulation (only applicable to DTINet) and the number of latent variables:
```TMTF.py DTINET --form_no 3 --f_size 32```

Set the stopping criterion and learning rate:
```TMTF.py E --epsilon 1e-5 --alpha 0.1```

Log the optimization process:
```TMTF.py NR --log```

The output is the predictions matrix $\hat{X}$ saved in the same directory as the `TMTF.py` file.


## Cite
``````