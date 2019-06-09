
import sklearn
import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.feature_selection import SelectKBest, chi2
from scipy.stats import multivariate_normal


def load_data(file_name):
    try:
        file = open(file_name, "r")
    except:
        print ("Error while opening file...")
        return -1
    target = []
    data_set = []

    for line in file:
        info = line.split(",")
        target.append(int(info[0]))
        data_set.append([float(a) for a in info[1:]])

    file.close()

    return data_set, target

#Load the data
data_set, target = load_data("data/wine.data.txt")
print ("data set:", data_set)

our_data = np.array(data_set,dtype=np.float32).reshape(((len(data_set), 13)))
print (our_data.shape)

our_target = np.array(target, dtype=np.int64).reshape((len(target)))

#For feature selection we have normalized all data...
mean_data = our_data.mean(axis=0)
std_data = our_data.std(axis=0)

max_data = our_data.max(axis=0)
min_data = our_data.min(axis=0)

diff = max_data - min_data
normalized_our_data = (our_data - min_data)
normalized_our_data = normalized_our_data/(diff)


print ("Normalized : ", normalized_our_data)
print ("Our data : ", our_data)

#feature selection
X_new = SelectKBest(chi2, k=2).fit_transform(normalized_our_data, our_target)
# We have chosen our important features,


#Seperating the data classes...
first_class = []
second_class = []
third_class = []

for i in range(len(our_target)):
    if our_target[i] == 1:
        first_class.append(X_new[i])
    elif our_target[i] == 2:
        second_class.append(X_new[i])
    elif our_target[i] == 3:
        third_class.append(X_new[i])



first_class = np.array(first_class, dtype=np.float32)
first_target = [1 for i in range(len(first_class))]

second_class = np.array(second_class, dtype=np.float32)
second_target = [2 for i in range(len(second_class))]

third_class = np.array(third_class, dtype=np.float32)
third_target = [3 for i in range(len(third_class))]


#Partitioning the given dataset as 80 percent tranining and 20 percent test for each class
first_train, first_test, first_y_train, first_y_test = train_test_split(first_class, first_target, test_size=0.33, random_state=42)
second_train, second_test, second_y_train, second_y_test = train_test_split(second_class, second_target, test_size=0.33, random_state=42)
third_train, third_test, third_y_train, third_y_test = train_test_split(third_class, third_target, test_size=0.33, random_state=42)

first_train = np.array(first_train, dtype=np.float32)
second_train = np.array(second_train, dtype=np.float32)
third_train = np.array(third_train, dtype=np.float32)


#Calculating mean and covariance for a given data set
def GetParameter(data_set):
  mean1 = np.mean(data_set, axis=0)

  cov1 = np.array([[0, 0],
                   [0, 0]], dtype=np.float32)

  for i in range(len(data_set)):
    fark = np.array([data_set[i] - mean1], dtype=np.float32)
    cov1 += np.matmul(fark.T, fark)

  cov1 /= len(data_set)

  return mean1, cov1

# Means and covariance for the 3 sets
first_mean, first_cov = GetParameter(first_train)
second_mean, second_cov = GetParameter(second_train)
third_mean, third_cov = GetParameter(third_train)

# Calculating Maximum Likelihood Estimation for 2-D Gaussian parameters
def TwoDGaussian (sample, mean, covarinace):
    fark = sample - mean
    exponential_part = np.exp((-1.0/2.0)*np.matmul(np.matmul(fark, np.linalg.inv(covarinace)), fark.T))
    beg = 1.0/(2*math.pi*math.sqrt(np.linalg.det(covarinace)))
    return beg*exponential_part


# Calculating prior probabilities for each class
prior_first = len(first_train)/(len(first_train)+len(second_train)+len(third_train))
prior_second = len(second_train)/(len(first_train)+len(second_train)+len(third_train))
prior_third = len(third_train)/(len(first_train)+len(second_train)+len(third_train))

# given mean and covariance calculating the probability of the sample on each class
# var = multivariate_normal(mean=first_mean, cov=first_cov)
# var2 = multivariate_normal(mean=second_mean, cov=second_cov)
# var3 = multivariate_normal(mean=third_mean, cov=third_cov)

correct_number = 0
false_number = 0


for i in range(len(first_test)):
    prob_first = TwoDGaussian(first_test[i], first_mean, first_cov) * prior_first
    prob_second = TwoDGaussian(first_test[i], second_mean, second_cov) * prior_first
    prob_third = TwoDGaussian(first_test[i], third_mean, third_cov) * prior_first
    #print(prob_first, prior_second, prob_third)
    if (prob_first > prob_second) and (prob_first > prob_third):
        correct_number += 1
    else:
        false_number += 1

for i in range(len(second_test)):
    prob_first = TwoDGaussian(second_test[i], first_mean, first_cov) * prior_second
    prob_second = TwoDGaussian(second_test[i], second_mean, second_cov) * prior_second
    prob_third = TwoDGaussian(second_test[i], third_mean, third_cov) * prior_second

    if (prob_second > prob_first) and (prob_second > prob_third):
        correct_number += 1
    else:
        false_number += 1


for i in range(len(third_test)):
    prob_first = TwoDGaussian(third_test[i], first_mean, first_cov) * prior_third
    prob_second = TwoDGaussian(third_test[i], second_mean, second_cov) * prior_third
    prob_third = TwoDGaussian(third_test[i], third_mean, third_cov) * prior_third

    if (prob_third > prob_first) and (prob_third > prob_second):
        correct_number += 1
    else:
        false_number += 1
print (correct_number)

print("Correct number : ", correct_number)
print("False number : ", false_number)
print("Accuracy : ", float(correct_number)/float(correct_number+false_number)*100)

exit()

#plotting the gaussian disributions
first_dim = np.linspace(0, 1, 40)
second_dim = np.linspace(0, 1, 40)
result = []

fig = plt.figure(2)
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

#Plotting Gaussian Density distribution for the first class
for a in range(len(first_dim)):
    for b in range(len(second_dim)):
        ax.scatter(first_dim[a], second_dim[b],TwoDGaussian([first_dim[a], second_dim[b]], first_mean, first_cov))

plt.title('First class Gaussian')
plt.axis('tight')
plt.show()

#Plotting Gaussian Density distribution for the second class
fig = plt.figure(3)
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

for a in range(len(first_dim)):
    for b in range(len(second_dim)):
        ax.scatter(first_dim[a], second_dim[b],TwoDGaussian([first_dim[a], second_dim[b]], second_mean, second_cov))

plt.title('Second class Gaussian')
plt.axis('tight')
plt.show()

#Plotting Gaussian Density distribution for the third class
fig = plt.figure(4)
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

for a in range(len(first_dim)):
    for b in range(len(second_dim)):
        ax.scatter(first_dim[a], second_dim[b],TwoDGaussian([first_dim[a], second_dim[b]], third_mean, third_cov))

plt.title('Third Gaussian...')
plt.axis('tight')
plt.show()

#Plotting Gaussian Density distribution for the all classes
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

for a in range(len(first_dim)):
    for b in range(len(second_dim)):
        ax.scatter(first_dim[a], second_dim[b],TwoDGaussian([first_dim[a], second_dim[b]], first_mean, first_cov))
        ax.scatter(first_dim[a], second_dim[b],TwoDGaussian([first_dim[a], second_dim[b]], second_mean, second_cov))
        ax.scatter(first_dim[a], second_dim[b],TwoDGaussian([first_dim[a], second_dim[b]], third_mean, third_cov))

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()






