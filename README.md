# MNIST Classification and Dimensionality Reduction

Comparing classifiers on MNIST, visualizing digits with PCA and t-SNE, and generating new images with a smooth bootstrap. This served as a final project for a graduate course in Statistical Methods in Machine Learning.

## Data Source

The data is the MNIST database of handwritten digits (LeCun, Cortes, and Burges), loaded through `keras.datasets.mnist`. It contains 60,000 training images and 10,000 test images, each a 28 x 28 grayscale image of a digit from 0 to 9. Pixel values were scaled to [0, 1] and each image was flattened to a 784-dimensional vector.

To keep computation manageable, the training set was reduced to a subset of the original 60,000 images. The full test set was used for evaluation.

## Part 1: Methodology 
