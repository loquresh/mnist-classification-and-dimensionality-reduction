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

The linear SVM uses a one-vs-rest scheme. Its regularization parameter `C` was tuned over a log-spaced grid from 0.001 to 1 with 5-fold cross-validation. The best value was `C = 0.03`, with a cross-validated accuracy of 0.9045. Accuracy fell off on both sides of this value, from underfitting at very small `C` and overfitting at large `C`. Values from 0.01 to 0.1 performed about equally well, so the exact choice matters little.

This is essentially identical to the tuned logistic regression (0.9047). Both models draw linear boundaries in pixel space, and both reach the same ceiling of about 0.905. A nonlinear model is needed to do better.

#### RBF kernel SVM

For the RBF kernel SVM, `C` was tuned over 1, 3, and 10 with 5-fold cross-validation, while `gamma` stayed at scikit-learn's data-based default (`"scale"`). The best value was `C = 10`, with a cross-validated accuracy of 0.9697. Accuracy was 0.9623 at `C = 1` and 0.9689 at `C = 3`, so the gains flatten out, and `C = 3` and `C = 10` are within the fold-to-fold variation of each other.

This is a large improvement over the linear models, which stopped at about 0.905. The RBF kernel lets the decision boundary curve in pixel space, and that flexibility is what separates the harder digit pairs. Scikit-learn's `SVC` trains one-vs-one classifiers internally, so this model uses a different multiclass scheme than the one-vs-rest linear models.

#### k-nearest neighbors

#### k-nearest neighbors

KNN classifies each image by a vote among its nearest training images, measured by Euclidean distance across all 784 pixels. The number of neighbors (1 to 15) and the weighting scheme (uniform or distance-weighted) were tuned with 5-fold cross-validation. The best setting was 3 neighbors with distance weighting, at a cross-validated accuracy of 0.9547. Five neighbors with distance weighting was nearly identical (0.9546).

Accuracy declined steadily as the number of neighbors grew, falling to about 0.94 at 15 neighbors, since larger neighborhoods include images from other digits. Distance weighting beat uniform weighting at every value above 1, by about 0.002. The two schemes are identical at 1 neighbor by construction.

The number of neighbors is a tuning parameter and has no connection to the number of digit classes. A common first guess of 10 neighbors would have scored about 0.944, a full percentage point below the tuned value.

### Model comparison

Each tuned model was scored once on the 10,000-image test set, which was not used at any point during tuning.

| Model | CV accuracy | Test accuracy | Prediction time (s) |
|---|---|---|---|
| Logistic regression | 0.9047 | 0.9102 | 0.5 |
| Linear SVM | 0.9045 | 0.9093 | 0.1 |
| RBF kernel SVM | 0.9697 | 0.9729 | 44.4 |
| KNN | 0.9547 | 0.9544 | 8.8 |

Test accuracy tracks cross-validated accuracy closely for all four models, so tuning did not overfit. The test scores run slightly above the cross-validated ones for three models, which is expected, since the final models were refit on all 15,000 training images while each cross-validation fold used 12,000.

The RBF kernel SVM is the most accurate model at 97.3%, followed by KNN at 95.4%. Logistic regression and the linear SVM are indistinguishable at about 91%. With 10,000 test images, differences of a few tenths of a percentage point fall within sampling noise, but the gaps between these three groups are much larger than that.

Prediction time moves in the opposite direction. The linear models predict the full test set in under a second, KNN takes about 9 seconds, and the RBF kernel SVM takes about 44. Prediction times were measured on a Colab CPU runtime.

![Confusion matrices for all four models](images/confusion_matrices.png)

The two linear models share the same weak spots. Digit 5 is the hardest, with recall of about 85%, and it is most often mistaken for 3 or 8. Digits 4 and 9 are confused in both directions: for logistic regression, 45 fours were predicted as nines and 43 nines as fours.

The RBF kernel SVM reduces these errors substantially. The 4/9 confusion falls to 16 in each direction, and recall on digit 5 rises to 96%. The curved decision boundary separates digit pairs that no straight line in pixel space can.

KNN has a different error profile. It is nearly perfect on 0 and 1 (about 99% and 99.6% recall) but weaker on 2 and 8 (about 92% each). Twos are most often predicted as 1, 7, or 0.
