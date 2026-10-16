# MNIST Classification and Dimensionality Reduction

Comparing classifiers on MNIST, visualizing digits with PCA and t-SNE, and generating new images with a smooth bootstrap. This served as a final project for a graduate course in Statistical Methods in Machine Learning.

## Data Source

The data is the MNIST database of handwritten digits (LeCun, Cortes, and Burges), loaded through `keras.datasets.mnist`. It contains 60,000 training images and 10,000 test images, each a 28 x 28 grayscale image of a digit from 0 to 9. Pixel values were scaled to [0, 1] and each image was flattened to a 784-dimensional vector.

To keep computation manageable, the training set was reduced to a subset of the original 60,000 images. The full test set was used for evaluation.

## Part 1: Methodology 

Let $f_1, \dots, f_K$ be densities on the real line and let $\alpha_1, \dots, \alpha_K$ be positive weights with $\sum_{i=1}^K \alpha_i = 1$. The mixture density is

$$
f_{\text{mix}}(x) = \sum_{i=1}^K \alpha_i f_i(x).
$$

Suppose we can draw samples from each $f_i$. To draw from $f_{\text{mix}}$:

1. Compute the cumulative weights $c_0 = 0$ and $c_i = \sum_{j=1}^{i} \alpha_j$ for $i = 1, \dots, K$, so that $c_K = 1$.
2. Draw $U \sim \text{Uniform}(0, 1)$.
3. Let $Z$ be the index $i$ such that $c_{i-1} < U \le c_i$.
4. Draw $X \sim f_Z$, using a fresh draw that is independent of $U$.
5. Return $X$.

### Why this works

The index $Z$ satisfies

$$
P(Z = i) = P(c_{i-1} < U \le c_i) = c_i - c_{i-1} = \alpha_i.
$$

Given $Z = i$, the sample $X$ has density $f_i$. By the law of total probability, the density of $X$ is

$$
f_X(x) = \sum_{i=1}^K P(Z = i)\, f_i(x) = \sum_{i=1}^K \alpha_i f_i(x) = f_{\text{mix}}(x).
$$

This is different from drawing $X_i \sim f_i$ for every $i$ and returning $\sum_i \alpha_i X_i$. That quantity is a weighted sum of random variables and does not follow the mixture distribution.

### Connection to the kernel density estimate

A Gaussian kernel density estimate with bandwidth $h$ fitted to points $x_1, \dots, x_n$ is a mixture with $\alpha_i = 1/n$ and $f_i = \mathcal{N}(x_i, h^2)$. Sampling from it means picking a data point uniformly at random and adding Gaussian noise. This is the sampler used in the data augmentation section.

## Data Analysis

### Preparing the data

Each MNIST image is a 28 x 28 grid of pixel values. I flattened each image into a vector of 784 values and scaled the values from the original 0 to 255 range down to 0 to 1, since SVMs and logistic regression perform better when features are on a small, consistent scale.

To keep computation manageable, trained on a stratified random subset of 15,000 images from the 60,000 available, which keeps the proportion of each digit the same as in the full set. The 10,000-image test set was left untouched during tuning and is used once at the end to compare the final models. All tuning used 5-fold stratified cross-validation within the training subset, with a fixed random seed so results are reproducible.

![Sample digits](images/sample_digits.png)

### Classification

Compared four classifiers on the flattened images: logistic regression, linear SVM, RBF kernel SVM, and k-nearest neighbors. The multiclass models use a one-vs-rest scheme, where a separate classifier is trained for each digit.

#### Logistic regression

With default settings, logistic regression reached a mean cross-validated accuracy of 0.903, with fold scores between 0.900 and 0.905.

Tuned the regularization strength `C` and the penalty type (L1 or L2) with a grid search. The best combination was `C = 0.5` with an L2 penalty, at a cross-validated accuracy of 0.905. That is only a small improvement over the default, and it falls within the fold-to-fold variation, so tuning did not change the model in a meaningful way.

The search results show that the top six settings were all within 0.002 of one another. Accuracy dropped at larger values of `C`, where the weaker penalty leads to overfitting. It also dropped at the smallest value with L1, where the penalty is strong enough to underfit. L2 held up better than L1 across the range of `C`, which is why I chose it.

| Setting | Mean CV accuracy |
|---|---|
| Default (`C = 1`, L2) | 0.903 |
| Tuned (`C = 0.5`, L2) | 0.905 |

Fitting time for the full grid search was about 9 minutes.

#### Linear SVM

*(In progress.)*

#### RBF kernel SVM

*(In progress.)*

#### k-nearest neighbors

*(In progress.)*

### Model comparison

*(To be added: test accuracy, confusion matrices, and runtime for all four models.)*
