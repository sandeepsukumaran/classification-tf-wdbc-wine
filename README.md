# CLASSIFICATION USING DEEP LEARNING

## ABOUT THE DATASETS
### Breast Cancer Wisconsin (Diagnostic) Data Set [WDBC]
source: [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
### Wine Dataset [WINE]
source: [https://archive.ics.uci.edu/ml/datasets/wine](https://archive.ics.uci.edu/ml/datasets/wine)

Custom modified versions of the datasets specified above were used. The custom datasets are provided as csv files. Training datasets have train suffix in their names.

## EXPERIMENTS

### SELECTION OF OPTIMIZER
With an initial simple model and the wdbc data-set, several initializers were compared using 5 fold cross validation. ADAM was selected as the best option over SGD, Adamax, Adagrad, and Adadelta. The parameters of ADAM itself was then tuned roughly using 5 fold cross validation and fine-tuned on different data-sets.

### SELECTION OF KERNEL INITIALIZER
Glorot Uniform (a.k.a. Xavier Uniform) was selected as the initialiser, over uniform, truncated normal, and LeCun initializers, based on empirical evidence from experiments. Bias initialisers are not used.

### SELECTION OF LEARNING RATE POLICY
Compared to a single stage optimiser,multi-stage optimisers were seen to perform better. Extending on this idea, Cyclical Learning Rates were implemented [based on ‘Cyclical Learning Rates for Training Neural Networks’ arXiv:1506.01186v6]. Experiments were carried out based on the simplest form suggested by the author - triangular learning rate, quantised at every 2 epochs.
Ultimately this was not seen to be the most helpful, possibly because the domain is not computer vision. Thus, it was scrapped in favour of multi-stage optimisers with decaying learning rates.

### SELECTION OF NEURON ACTIVATION
Initial testing was done using ReLU as the activation of all nodes. However, constant loss and accuracy values were noticed from very early epochs, indicative of the ’dead ReLU’ problem. Thus, Leaky ReLUs were used instead. Further experiments showed that greater the slope permitted by the Leaky ReLUs, the better the model performed. Extending on this trend, linear activation was selected throughout.

### SELECTION OF LOSS FUNCTION
As of Tensorflow 1.8 (which was used throughout this project’s lifetime), the default implementation of _categorical\_cross\_entropy_ in tensorflow.python.keras is buggy. The call to the function assumes that the actual input argument to the function is not logits, i.e. that a probability vector is provided. Since this may not always be the case, softmax is required (to avoid NaN loss). Rather than use a separate softmax layer, a custom loss function was defined which internally calls _tf.nn.softmax\_cross\_entropy\_with\_logits\_v2_, the function called by _categorical\_cross\_entropy_ if the flag variable for from logits was set.

The models used in ``small” scenarios have a simpler architecture since these were found to suffice, due to dropouts and regularisation.

Stable accuracy results were observed, as detailed below:

| Program Name | Accuracy Range |
| :----------: | :------------: |
| WDBC         | 0.91 - 0.95    |
| WDBC-small   | 0.91 - 0.94    |
| WINE-small   | 0.93 - 0.94    |
| WINE         | 0.95 - 0.96    |

These results are without normalization of data. With normalization, accuracies of 0.96 are obtained consistently across all cases.

