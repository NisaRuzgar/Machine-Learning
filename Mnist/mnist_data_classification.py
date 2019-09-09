
# NISA PINAR RUZGAR / 220201050

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt


# HYPERPARAMETERS
input_size = 784
output_size = 10
layer_info = [784, 80, 32, 10]
BATCH_SIZE = 10000
learning_rate = 0.001
number_of_epochs = 120
path = "./mnist" #please use relative path like this


# we are creating weights and bias matrices

#layer delta stores the weight changes in backpropagation part...
# layer outputs stores the output values of nn layer's for backpropagation
# layer weights stores the weights of layers...

# layer_delta is a dictionary. It has to key parameters. "w" holds the weight matrices of each layer
# "b" holds the bias of each layer...
# Each key holds an array for storing weights of layers...
# "w": [first layer weight matrix, second hidden layer weight matrix .....]
# "b": [first layer bias matrix, second hidden layer bias matrix ....]

layer_delta = {"w" : [0 for i in range(len(layer_info)-1)],
"b" : [0 for j in range(len(layer_info)-1)]}
layer_outputs = [0 for k in range(len(layer_info)-1)]

# layer_weights is an array..
# for each layer it stores a dictionary that holds bias and weight matrices...
# layer_weights = [ { "w": first layer weight matrix, "b": first layer bias matrix}, {"w": second hidden layer weight matrix, "b": second hidden layer bias matrix}....]
layer_weights = [{} for l in range(len(layer_info)-1)]

# We are creating weights and biases for each layer...
for layer_index in range(len(layer_info)-1):
    # Weights are randomly sampled between [-1, 1]
    w = np.random.uniform(-1.0,1.0,(layer_info[layer_index], layer_info[layer_index+1]))
    b = np.random.uniform(-1.0,1.0,(1, layer_info[layer_index+1]))

    layer_weights[layer_index]["w"] = w
    layer_weights[layer_index]["b"] = b



def activation_function(layer):
    return 1.0 / (1.0 + np.exp(-layer))


def derivation_of_activation_function(signal):
    return signal*(1-signal)

def loss_function(true_labels, probabilities):
    return -np.sum(true_labels * np.log(probabilities))


def softmax(layer):
    scorematexp = np.exp(layer)
    return scorematexp / np.sum(scorematexp, axis=1).reshape((scorematexp.shape[0], 1))
# softmax is used to turn activations into probability distribution


def derivation_of_loss_function(true_labels, probabilities):
    return true_labels-probabilities
    pass
# the derivation should be with respect to the output neurons


def forward_pass(data):
    """
        For the given input, it calculates all layer outputs...
        :param input: the input array to nn
        :return: output layer's values
        nn weights and layer_outputs arrays are included globally...

        example:
        matrix calculations (for 4 layer nn):

        input: input matrix
        w1: weight matrix of first hidden layer
        b1: bias matrix of first hidden layer

        o1: output for first hidden layer (sigmoid is not applied)
        s1: sigmoid(o1)

        o1 = input.w1 + b1
        s1 = sigmoid(o1)

        o2 = s1.w2 + b2
        s2 = sigmoid(o2)

        o3 = s2.w3+b3

        o3(output) = softmax(o3)

        for the output layer we are applying softmax...

        biases are row vectors. To add them with weight matrices they need to be multiplied via 1 vectors.

        """

    global layer_weights, layer_outputs

    # 1 vector for all bias weights...
    ones = np.ones(len(data)).reshape(1, len(data)).T

    # it stores the previous layer output
    prev_layer_input = data
    # we are calculating outputs of each layer except the output layer...
    for layer_index in range(len(layer_info) - 1 - 1):
      # calculations are done as explained above...
      # layer outputs are stored in global array...
      layer_outputs[layer_index] = activation_function(
        np.matmul(prev_layer_input, layer_weights[layer_index]["w"]) + np.matmul(ones,
                                             layer_weights[layer_index]["b"]))
      prev_layer_input = layer_outputs[layer_index]

    # for output we are using softmax
    layer_outputs[-1] = softmax(
      np.matmul(prev_layer_input, layer_weights[-1]["w"]) + np.matmul(ones, layer_weights[-1]["b"]))
    output = layer_outputs[-1]
    # returning output layer values
    return output, layer_outputs



# [hidden_layers] is not an argument, but it is up to you how many hidden layers to implement.
# so replace it with your desired hidden layers 
def backward_pass(input_layer, layer_outputs , output_layer, loss):
    """
       It adjusts nn weights for the given training data...

       Matrix operations (for 4 layer):
       . -> matrix multiplication
       * -> hadamard(pairwise mutliplication)
       ()^T -> transpose operation


       output_error: target - nn output(softmax)
       w2_delta = (s2)^T.output_error
       b2_delta = (1 vector)^T.output_error

       layer2_error = (output_error.(w2)^T)*s2*(1-s2)
       w1_delta = (s1)^T.layer2_error
       b1_delta = (1 vector)^T.layer2_error

       layer1_error = (layer2_error.(w1)^T)*s1*(1-s1)
       w1_delta = (input)^T.layer1_error
       b1_delta = (1 vector)^T.layer1_error

       w1 = w1+learning_rate*w1_delta
       b1 = b1+learning_rate*b1_delta

       w2 = w2 + learning_rate * w2_delta
       b2 = b2 + learning_rate * b2_delta


       w3 = w3 + learning_rate * w3_delta
       b3 = b3 + learning_rate * b3_delta

       :param input:
       :param target:
       :return:
       """
    global layer_weights, layer_delta
    error = loss
    ones = np.ones(len(input_layer)).reshape((1, len(input_layer)))

    layer_error = error

    # layer weight changes are handled as shown above...
    for layer_index in range(len(layer_info) - 1 - 1, 0, -1):
      layer_delta["b"][layer_index] = learning_rate * np.matmul(ones, layer_error)
      layer_delta["w"][layer_index] = learning_rate * np.matmul(layer_outputs[layer_index - 1].T, layer_error)
      layer_error = np.matmul(layer_error, np.transpose(layer_weights[layer_index]["w"])) * (
      layer_outputs[layer_index - 1]) * (1 - layer_outputs[layer_index - 1])

    # For first layer input is used, we haven't included it in layer outputs so we are handling it seperately...
    delta_b1 = learning_rate * np.matmul(ones, layer_error)
    delta_a1 = learning_rate * np.matmul(input_layer.T, layer_error)

    layer_delta["b"][0] = delta_b1
    layer_delta["w"][0] = delta_a1

    # apply weight changes to layers...
    for i in range(len(layer_info) - 1):
      layer_weights[i]["w"] += layer_delta["w"][i]
    for i in range(len(layer_info) - 1):
      layer_weights[i]["b"] += layer_delta["b"][i]


epoch_step = []
train_cross_entropy_set = []
validation_cross_entropy_set = []
train_accuracy_set = []
validation_accuracy_set = []

def train(train_data, train_labels, valid_data, valid_labels):
    for epoch in range(number_of_epochs):
      index = 0
      # Total train cross entropy loss
      losses = 0

      # Same thing about [hidden_layers] mentioned above is valid here also

      for batch_index in range(len(train_data)//BATCH_SIZE):
        # print("batch index : ", batch_index)
        data = train_data[batch_index*BATCH_SIZE: (batch_index+1)*BATCH_SIZE]
        labels = train_labels[batch_index*BATCH_SIZE: (batch_index+1)*BATCH_SIZE]
        # for data, labels in zip(train_data, train_labels):
        # print("data : ", data, " label : ", labels)

        predictions, layer_outputs = forward_pass(data)
        loss_signals = derivation_of_loss_function(labels, predictions)
        backward_pass(data, layer_outputs, predictions, loss_signals)
        loss = loss_function(labels, predictions)
        losses += loss
        if index%2000 == 0: # at each 2000th sample, we run validation set to see our model's improvements
          accuracy, loss = test(valid_data, valid_labels)
          print("Epoch= "+str(epoch)+", Coverage= %"+ str(100*(index/len(train_data))) + ", Accuracy= "+ str(accuracy) + ", Loss= " + str(loss))

        index += BATCH_SIZE


      train_prediction, _ = forward_pass(train_data)
      validation_prediction, _ = forward_pass(valid_data)

      train_accuracy = np.sum(np.equal(np.argmax(train_prediction, axis=1), np.argmax(train_labels, axis=1)))/len(train_labels)*100
      validation_accuracy = np.sum(np.equal(np.argmax(validation_prediction, axis=1), np.argmax(valid_labels, axis=1)))/len(valid_labels)*100

      validation_cross_entropy = loss_function(valid_labels, validation_prediction)


      epoch_step.append(epoch)
      train_cross_entropy_set.append(losses/len(train_data))
      train_accuracy_set.append(train_accuracy)

      validation_accuracy_set.append(validation_accuracy)
      validation_cross_entropy_set.append(validation_cross_entropy/len(valid_data))
    return losses


def test(test_data, test_labels):
    avg_loss = 0
    predictions = []
    labels = []

    #for data, label in zip(test_data, test_labels): # Turns through all data

    predictions, _ = forward_pass(test_data)
    #predictions.append(prediction)
    #labels.append(label)
    avg_loss += np.sum(loss_function(test_labels, predictions))

    # Maximum likelihood is used to determine which label is predicted, highest prob. is the prediction
    # And turn predictions into one-hot encoded


    #one_hot_predictions = np.zeros(shape=(len(predictions), output_size))
    #for i in range(len(predictions)):
    #	one_hot_predictions[i][np.argmax(predictions[i])] = 1
    # predictions = one_hot_predictions

    accuracy_score = accuracy(test_labels, predictions)

    return accuracy_score,  avg_loss / len(test_data)


def accuracy(true_labels, predictions):
    true_pred = 0

    for i in range(len(predictions)):
      if np.argmax(predictions[i]) == np.argmax(true_labels[i]): # if 1 is in same index with ground truth
        true_pred += 1

    return true_pred / len(predictions)


if __name__ == "__main__":
    mndata = MNIST(path)
    train_x, train_y = mndata.load_training()
    test_x, test_y = mndata.load_testing()

    # converting numpy array
    train_x = np.array(train_x, dtype=np.float32)
    test_x = np.array(test_x, dtype=np.float32)

    train_y = np.array(train_y, dtype=np.int32)
    test_y = np.array(test_y, dtype=np.int32)


    # normalizing test and train sets...
    train_mean = train_x.mean()
    train_min = train_x.min()
    train_max = train_x.max()

    train_x = (train_x-train_mean)/(train_max-train_min)
    test_x = (test_x-train_mean)/(train_max-train_min)

    print("train size : ", train_x.shape)
    print("test size : ", test_x.shape)


    # creating one-hot vector notation of labels. (Labels are given numeric in MNIST)
    new_train_y = np.zeros(shape=(len(train_y), output_size))
    new_test_y = np.zeros(shape=(len(test_y), output_size))

    for i in range(len(train_y)):
      new_train_y[i][train_y[i]] = 1

    for i in range(len(test_y)):
      new_test_y[i][test_y[i]] = 1

    train_y = new_train_y
    test_y = new_test_y

    # Training and validation split. (%80-%20)
    valid_x = np.asarray(train_x[int(0.8*len(train_x)):-1])
    valid_y = np.asarray(train_y[int(0.8*len(train_y)):-1])
    train_x = np.asarray(train_x[0:int(0.8*len(train_x))])
    train_y = np.asarray(train_y[0:int(0.8*len(train_y))])

    print("valid_X : ", valid_x.shape)
    print("valid_y : ", valid_y.shape)

    train(train_x, train_y, valid_x, valid_y)
    print("Test Scores:")
    print(test(test_x, test_y))

    # We are displaying test and train cross entropy losses
    # For appropriate graphing, we are displaying mean normalized cross entropy
    plt.figure(200)
    plt.ylabel("Mean Cross Entropy Loss")
    plt.xlabel("Epoch Number")
    plt.title("Train and Validation Set Mean Cross Entropy Loss")
    plt.plot(epoch_step, train_cross_entropy_set, 'r', label="Train Loss")
    plt.plot(epoch_step, validation_cross_entropy_set, 'b', label="Validation Loss")
    plt.legend(loc='upper right')

    # We are displaying the accuracy over train and test set for each epoch...
    plt.figure(201)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch Number")
    plt.title("Train and Validation Set Accuracy (%)")
    plt.plot(epoch_step, train_accuracy_set, 'r', label="Train Accuracy")
    plt.plot(epoch_step, validation_accuracy_set, 'b', label="Validation Accuracy")
    plt.legend(loc='upper right')

    plt.show()



