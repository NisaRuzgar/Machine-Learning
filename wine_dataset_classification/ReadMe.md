Calculation of Discriminant Functions for 2-D Multivariate Normal Density on the Wine Dataset.

The Wine dataset has three types of classes.
As it's 2-D Gaussian and the dataset contains 13 attributes,I have done feature selection first.
I have partitioned the given dataset as 80% training and 20% test. 
Using the training partition, for each class, ý have done Maximum Likelihood Estimation (MLE) for 2-D Gaussian parameters: Mean and covariance.
Plotted each 2-D Gaussian.
To classify the instances in the test set partition, I have calculated unnormalized posteriors depending on the Gaussians and calculated the accuracy.
