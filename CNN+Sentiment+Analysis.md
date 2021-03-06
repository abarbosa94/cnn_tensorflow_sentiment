

```python
import numpy as np
import re
import itertools
from collections import Counter
from tensorflow.contrib import learn
import unittests as tests
import tensorflow as tf
```

# Funcoes Auxiliares (tiradas do paper)


```python
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def batch_iter(data, batch_size, num_batches_per_epoch, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    #all data
    data_size = len(data)
    #each block has 64 data input, resulting in 150 blocks data
    #num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            print('Epoch {:>2}, Sentence Batch {}:  '.format(epoch + 1, batch_num), end='')
            #150 blocks of data per epoch
            yield shuffled_data[start_index:end_index]
```

# Data explore


```python
import random

data,labels = load_data_and_labels('data/rt-polaritydata/rt-polarity.pos','data/rt-polaritydata/rt-polarity.neg')

print (len(data)) #todos os reviews, cada elemento eh um review
print (len(labels)) #todos os sentimentos

```

    10662
    10662



```python
(data[0], labels[0]) #first review and its label
```




    ("the rock is destined to be the 21st century 's new conan and that he 's going to make a splash even greater than arnold schwarzenegger , jean claud van damme or steven segal",
     array([0, 1]))



# Pre processing
- Build vocabulary


```python
max_review_size = max([len(x.split(" ")) for x in data])

#Maps documents to sequences of word ids.
vocab_processor = learn.preprocessing.VocabularyProcessor(max_review_size)
x = np.array(list(vocab_processor.fit_transform(data)))#aqui que ocorre o mapeamento
(x[0], data[0])
```




    (array([ 1,  2,  3,  4,  5,  6,  1,  7,  8,  9, 10, 11, 12, 13, 14,  9, 15,
             5, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0]),
     "the rock is destined to be the 21st century 's new conan and that he 's going to make a splash even greater than arnold schwarzenegger , jean claud van damme or steven segal")




```python
max_review_size
```




    56



- Randomly shuffle data


```python
#Shuffle Data
np.random.seed(10) #for debugging, garante que os numeros aleatorios gerados sempre sejam os mesmos
shuffle_indices = np.random.permutation(np.arange(len(labels)))
shuffle_indices
```




    array([ 7359,  5573, 10180, ...,  1344,  7293,  1289])




```python
#e.g.: x[7359] == x_shuffled[0]
x_shuffled = x[shuffle_indices] 
y_shuffled = labels[shuffle_indices]


```

 - Train/Validation split


```python
val_percentage = .1
val_sample_index = -1 * int(val_percentage * float(len(labels)))
```


```python
x_train, x_val = x_shuffled[:val_sample_index], x_shuffled[val_sample_index:]
y_train, y_val = y_shuffled[:val_sample_index], y_shuffled[val_sample_index:]
```


```python
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_val)))
```

    Vocabulary Size: 18758
    Train/Dev split: 9596/1066


# Build CNN

- Inputs and Labels instances


```python
def neural_net_sentence_input(sentence_size):
    """
    Return a Tensor for a batch of image input
    : sentence_size: Size of the sentence with the biggest len
    : return: Tensor for sentences input.
    Remeber: all sentences were padded to get the max len
    """
    return tf.placeholder(tf.int32, shape=[None,sentence_size],name='input_x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, shape=[None,n_classes],name='input_y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, shape=None,name='keep_prob')


"""
UNIT TESTS
"""
tf.reset_default_graph()
tests.test_nn_sentence_inputs(neural_net_sentence_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)
```

    Sentence Input Tests Passed.
    Label Input Tests Passed.
    Keep Prob Tests Passed.


- Load Pre Trained Word2Vec Model


```python
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = None
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 


```

    2017-06-16 20:57:08,092 : INFO : loading projection weights from GoogleNews-vectors-negative300.bin
    2017-06-16 20:58:00,644 : INFO : loaded (3000000, 300) matrix from GoogleNews-vectors-negative300.bin


- Store only the words that exists in our vocab 


```python
# Remove previous weights, bias, inputs, etc..

tf.reset_default_graph()
vocab_size = len(vocab_processor.vocabulary_)
W = tf.Variable(initial_value=tf.random_uniform([vocab_size, 300], -1.0, 1.0),name="K")
if(model):
    T = np.random.rand(vocab_size, 300)
vocab_dict = vocab_processor.vocabulary_._mapping
for word,idx in vocab_dict.items():
    if word in model:
        T[idx] = model[word]
    else:
        T[idx] = np.random.uniform(low=-0.25, high=0.25, size=(300,))
#save memory
del model
```

- Embedding Layer


```python
def embedding_creation(x_tensor,vocab_size,embedding_size):
    embedded_chars = tf.nn.embedding_lookup(W, x_tensor)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
    
    return embedded_chars_expanded


tests.test_embed(embedding_creation)
```

    Tests Passed


- Convolution Layer


```python
def conv2d_maxpool(x_tensor, num_filters, filter_size):
    """
    return: A tensor that represents convolution and max pooling of x_tensor
    """
    embbeding_size = int(x_tensor.shape[2])
    filter_shape = [filter_size,embbeding_size, 1, num_filters]
    
    weights = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1), name="W")
    bias = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
    """
    Strides controls how the filter convolves around the input
    As we want to go each word per time, everything will have size one
    As we apply conv layers, we could pad the image to preserve the dimension 
    (and try to extract more level features)
    Because we are only dealing with words, this would not be necessary. This is known as narrow convolution
    
    Conv gives us an output of shape [1, sequence_length - filter_size + 1, 1, 1] - There is a formula to discover that
    
    """
    conv = tf.nn.conv2d(x_tensor, weights, strides=[1, 1, 1, 1], padding='VALID')

    conv = tf.nn.bias_add(conv, bias)
    #add non linearity
    h = tf.nn.relu(conv, name="relu")
    sequence_length = int(x_tensor.shape[1])
    conv_output = [1, sequence_length - filter_size + 1, 1, 1]
    
    #Maxpooling over the outputs
    #this will heaturn a tensor of shape [batch_size, 1, 1, num_filters] 
    #which is essencialy a feature vector where the last dimension correspond to features
    #Stride have this size basically because of the same logic applied before
    pooled = tf.nn.max_pool(h, ksize=conv_output,
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name='pool') 
    
    return pooled


tests.test_con_pool(conv2d_maxpool)
```

    Tests Passed


- Apply different filters


```python
def apply_conv_filters(x_tensor,filter_sizes,num_filters):
# Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-{}".format(filter_size)):
            pooled = conv2d_maxpool(x_tensor, num_filters, filter_size)
            pooled_outputs.append(pooled)     
    num_filters_total = num_filters * len(filter_sizes)
    #concat -> sum(Daxis(i)) where Daxis is Dimension axis (in our case is the third one)
    h_pool = tf.concat(pooled_outputs, 3)
    return h_pool

tests.test_apply_filters(apply_conv_filters,conv2d_maxpool)
```

    Tests Passed


- Flatten Layer

The flatten function to change the dimension of x_tensor from a 4-D tensor to a 2-D tensor. The output should be the shape (Batch Size, Flattened Features Size).


```python
def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    #This is a general flatten function
    flat = x_tensor.shape[1]*x_tensor.shape[2]*x_tensor.shape[3]
    return tf.reshape(x_tensor,[-1,int(flat)])



tests.test_flatten(flatten)
```

    Tests Passed



```python
def output(x_tensor,num_classes):
    num_filters_total = int(x_tensor.shape[1])
    W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

    scores = tf.nn.xw_plus_b(x_tensor, W, b, name="scores")
    return scores, tf.nn.l2_loss(W), tf.nn.l2_loss(b)

```

- Convolutional Network


```python
def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    
    vocab_size = len(vocab_processor.vocabulary_)
    embed_dim = 300
    with tf.name_scope("embedding"):
        embbed_layer = embedding_creation(x,vocab_size,embed_dim)
    
    num_filters = 128
    filter_sizes = [3,4,5]
    conv_layer = apply_conv_filters(embbed_layer,filter_sizes,num_filters)

    

    flat_layer = flatten(conv_layer)
    
    with tf.name_scope("dropout"):
        dropout =  tf.nn.dropout(flat_layer, keep_prob)
        
    with tf.name_scope("output"):
        num_classes = 2
        output_layer, l2_w, l2_b = output(dropout, num_classes)

    
    return output_layer, l2_w, l2_b
```


```python
#Regularization parameters
l2_loss = tf.constant(0.0)
l2_reg_lambda = 1.0


# Inputs
x_input = neural_net_sentence_input(56) #sequence_length
y_input = neural_net_label_input(2) #positive or negative
keep_prob = neural_net_keep_prob_input()

# Model
logits, l2_w, l2_b = conv_net(x_input, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_input))
l2_loss += l2_w
l2_loss += l2_b
cost =  cost + l2_reg_lambda * l2_loss
#optimizer = tf.train.AdamOptimizer().minimize(cost) - Other option for the optmizer, but got less validation acc
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-3)
grads_and_vars = optimizer.compute_gradients(cost)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


tests.test_conv_net(conv_net)
```

    Neural Network Built!


# Training Process


```python
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    
    session.run(optimizer, feed_dict={
            x_input: feature_batch,
            y_input: label_batch,
            keep_prob: keep_probability,
            })


tests.test_train_nn(train_neural_network)
```

    Tests Passed


# Print statistics


```python
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    loss,acc = session.run([cost,accuracy],feed_dict={
            x_input: feature_batch,
            y_input: label_batch,
            keep_prob: 1.})
    
    
    print('Loss: {:>10.4f} Training Accuracy: {:.6f}'.format(loss,acc))
```


```python
def print_validation_stats(session):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    
    valid_acc = session.run(accuracy, feed_dict={
        x_input: x_val,
        y_input: y_val,
        keep_prob: 1.})
    
    print('Validation Accuracy: {:.6f}'.format(valid_acc))
```

# Hyperparameters

- Just for a Single Batch


```python
epochs = 12
batch_size = 64
keep_probability =  0.5
num_batches_per_epoch = 1
```

# Training on a Single Batch


```python
print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    if(T.any):
        sess.run(W.assign(T))
    # Generate single batches
    batches = batch_iter(list(zip(x_train, y_train)), batch_size, 1, epochs, shuffle=False)
    # Training cycle
    for batch in batches:
        batch_features, batch_labels = zip(*batch)
        train_neural_network(sess, train_op, keep_probability, batch_features, batch_labels)
        print_stats(sess, batch_features, batch_labels, cost, accuracy)
    print("#######VALIDATION STATS#######")
    print_validation_stats(sess)
```

    Checking the Training on a Single Batch...
    Epoch  1, Sentence Batch 0:  Loss:     3.3301 Training Accuracy: 0.687500
    Epoch  2, Sentence Batch 0:  Loss:     3.1099 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 0:  Loss:     2.9558 Training Accuracy: 0.984375
    Epoch  4, Sentence Batch 0:  Loss:     2.8493 Training Accuracy: 1.000000
    Epoch  5, Sentence Batch 0:  Loss:     2.7624 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 0:  Loss:     2.6868 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 0:  Loss:     2.6201 Training Accuracy: 1.000000
    Epoch  8, Sentence Batch 0:  Loss:     2.5610 Training Accuracy: 1.000000
    Epoch  9, Sentence Batch 0:  Loss:     2.5060 Training Accuracy: 1.000000
    Epoch 10, Sentence Batch 0:  Loss:     2.4552 Training Accuracy: 1.000000
    Epoch 11, Sentence Batch 0:  Loss:     2.4080 Training Accuracy: 1.000000
    Epoch 12, Sentence Batch 0:  Loss:     2.3628 Training Accuracy: 1.000000
    #######VALIDATION STATS#######
    Validation Accuracy: 0.561914


- Update Hyperparameters for full training


```python
epochs = 7
batch_size = 64
keep_probability =  0.3
num_batches_per_epoch = int((len(list(zip(x_train, y_train)))-1)/batch_size) + 1
```


```python
print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    if(T.any):
        sess.run(W.assign(T))
    # Generate single batches
    batches = batch_iter(list(zip(x_train, y_train)), batch_size, num_batches_per_epoch, epochs, shuffle=True)
    # Training cycle
    i = 0 
    for batch in batches:
        if(i%100 == 0):
            print_validation_stats(sess)
            i = 0
        batch_features, batch_labels = zip(*batch)
        train_neural_network(sess, train_op, keep_probability, batch_features, batch_labels)
        print_stats(sess, batch_features, batch_labels, cost, accuracy)
        i+=1
        
    print("#######VALIDATION STATS#######")
    print_validation_stats(sess)
```

    Training...
    Epoch  1, Sentence Batch 0:  Validation Accuracy: 0.533771
    Loss:     3.5203 Training Accuracy: 0.718750
    Epoch  1, Sentence Batch 1:  Loss:     3.6582 Training Accuracy: 0.546875
    Epoch  1, Sentence Batch 2:  Loss:     3.7570 Training Accuracy: 0.500000
    Epoch  1, Sentence Batch 3:  Loss:     3.5578 Training Accuracy: 0.656250
    Epoch  1, Sentence Batch 4:  Loss:     3.5281 Training Accuracy: 0.609375
    Epoch  1, Sentence Batch 5:  Loss:     3.4202 Training Accuracy: 0.703125
    Epoch  1, Sentence Batch 6:  Loss:     3.4860 Training Accuracy: 0.609375
    Epoch  1, Sentence Batch 7:  Loss:     3.3356 Training Accuracy: 0.703125
    Epoch  1, Sentence Batch 8:  Loss:     3.3841 Training Accuracy: 0.625000
    Epoch  1, Sentence Batch 9:  Loss:     3.3328 Training Accuracy: 0.593750
    Epoch  1, Sentence Batch 10:  Loss:     3.2570 Training Accuracy: 0.640625
    Epoch  1, Sentence Batch 11:  Loss:     3.2412 Training Accuracy: 0.703125
    Epoch  1, Sentence Batch 12:  Loss:     3.3232 Training Accuracy: 0.578125
    Epoch  1, Sentence Batch 13:  Loss:     3.1776 Training Accuracy: 0.671875
    Epoch  1, Sentence Batch 14:  Loss:     3.1245 Training Accuracy: 0.734375
    Epoch  1, Sentence Batch 15:  Loss:     3.0831 Training Accuracy: 0.718750
    Epoch  1, Sentence Batch 16:  Loss:     3.0391 Training Accuracy: 0.750000
    Epoch  1, Sentence Batch 17:  Loss:     2.9785 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 18:  Loss:     3.0201 Training Accuracy: 0.671875
    Epoch  1, Sentence Batch 19:  Loss:     2.9341 Training Accuracy: 0.734375
    Epoch  1, Sentence Batch 20:  Loss:     2.9810 Training Accuracy: 0.609375
    Epoch  1, Sentence Batch 21:  Loss:     2.9178 Training Accuracy: 0.687500
    Epoch  1, Sentence Batch 22:  Loss:     2.9026 Training Accuracy: 0.750000
    Epoch  1, Sentence Batch 23:  Loss:     2.7769 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 24:  Loss:     2.8199 Training Accuracy: 0.703125
    Epoch  1, Sentence Batch 25:  Loss:     2.7586 Training Accuracy: 0.796875
    Epoch  1, Sentence Batch 26:  Loss:     2.8036 Training Accuracy: 0.671875
    Epoch  1, Sentence Batch 27:  Loss:     2.7191 Training Accuracy: 0.750000
    Epoch  1, Sentence Batch 28:  Loss:     2.7337 Training Accuracy: 0.703125
    Epoch  1, Sentence Batch 29:  Loss:     2.5988 Training Accuracy: 0.781250
    Epoch  1, Sentence Batch 30:  Loss:     2.6216 Training Accuracy: 0.671875
    Epoch  1, Sentence Batch 31:  Loss:     2.6411 Training Accuracy: 0.703125
    Epoch  1, Sentence Batch 32:  Loss:     2.6092 Training Accuracy: 0.656250
    Epoch  1, Sentence Batch 33:  Loss:     2.5889 Training Accuracy: 0.656250
    Epoch  1, Sentence Batch 34:  Loss:     2.5107 Training Accuracy: 0.734375
    Epoch  1, Sentence Batch 35:  Loss:     2.3852 Training Accuracy: 0.843750
    Epoch  1, Sentence Batch 36:  Loss:     2.4920 Training Accuracy: 0.640625
    Epoch  1, Sentence Batch 37:  Loss:     2.4026 Training Accuracy: 0.718750
    Epoch  1, Sentence Batch 38:  Loss:     2.5242 Training Accuracy: 0.578125
    Epoch  1, Sentence Batch 39:  Loss:     2.4172 Training Accuracy: 0.671875
    Epoch  1, Sentence Batch 40:  Loss:     2.4380 Training Accuracy: 0.671875
    Epoch  1, Sentence Batch 41:  Loss:     2.4137 Training Accuracy: 0.656250
    Epoch  1, Sentence Batch 42:  Loss:     2.3489 Training Accuracy: 0.671875
    Epoch  1, Sentence Batch 43:  Loss:     2.3212 Training Accuracy: 0.703125
    Epoch  1, Sentence Batch 44:  Loss:     2.2513 Training Accuracy: 0.687500
    Epoch  1, Sentence Batch 45:  Loss:     2.2261 Training Accuracy: 0.781250
    Epoch  1, Sentence Batch 46:  Loss:     2.2173 Training Accuracy: 0.640625
    Epoch  1, Sentence Batch 47:  Loss:     2.1938 Training Accuracy: 0.671875
    Epoch  1, Sentence Batch 48:  Loss:     2.2825 Training Accuracy: 0.578125
    Epoch  1, Sentence Batch 49:  Loss:     2.1625 Training Accuracy: 0.656250
    Epoch  1, Sentence Batch 50:  Loss:     2.1615 Training Accuracy: 0.656250
    Epoch  1, Sentence Batch 51:  Loss:     2.1820 Training Accuracy: 0.625000
    Epoch  1, Sentence Batch 52:  Loss:     2.1010 Training Accuracy: 0.671875
    Epoch  1, Sentence Batch 53:  Loss:     2.0360 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 54:  Loss:     2.0585 Training Accuracy: 0.640625
    Epoch  1, Sentence Batch 55:  Loss:     2.0330 Training Accuracy: 0.718750
    Epoch  1, Sentence Batch 56:  Loss:     2.0350 Training Accuracy: 0.703125
    Epoch  1, Sentence Batch 57:  Loss:     1.9573 Training Accuracy: 0.718750
    Epoch  1, Sentence Batch 58:  Loss:     1.9070 Training Accuracy: 0.796875
    Epoch  1, Sentence Batch 59:  Loss:     1.9392 Training Accuracy: 0.734375
    Epoch  1, Sentence Batch 60:  Loss:     1.8227 Training Accuracy: 0.843750
    Epoch  1, Sentence Batch 61:  Loss:     1.8809 Training Accuracy: 0.734375
    Epoch  1, Sentence Batch 62:  Loss:     1.8274 Training Accuracy: 0.781250
    Epoch  1, Sentence Batch 63:  Loss:     1.8374 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 64:  Loss:     1.8467 Training Accuracy: 0.687500
    Epoch  1, Sentence Batch 65:  Loss:     1.8184 Training Accuracy: 0.781250
    Epoch  1, Sentence Batch 66:  Loss:     1.8128 Training Accuracy: 0.703125
    Epoch  1, Sentence Batch 67:  Loss:     1.8240 Training Accuracy: 0.671875
    Epoch  1, Sentence Batch 68:  Loss:     1.7071 Training Accuracy: 0.796875
    Epoch  1, Sentence Batch 69:  Loss:     1.8092 Training Accuracy: 0.656250
    Epoch  1, Sentence Batch 70:  Loss:     1.7038 Training Accuracy: 0.796875
    Epoch  1, Sentence Batch 71:  Loss:     1.6927 Training Accuracy: 0.796875
    Epoch  1, Sentence Batch 72:  Loss:     1.7564 Training Accuracy: 0.703125
    Epoch  1, Sentence Batch 73:  Loss:     1.6449 Training Accuracy: 0.718750
    Epoch  1, Sentence Batch 74:  Loss:     1.6783 Training Accuracy: 0.718750
    Epoch  1, Sentence Batch 75:  Loss:     1.6045 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 76:  Loss:     1.6060 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 77:  Loss:     1.6417 Training Accuracy: 0.671875
    Epoch  1, Sentence Batch 78:  Loss:     1.5869 Training Accuracy: 0.750000
    Epoch  1, Sentence Batch 79:  Loss:     1.5127 Training Accuracy: 0.828125
    Epoch  1, Sentence Batch 80:  Loss:     1.5474 Training Accuracy: 0.781250
    Epoch  1, Sentence Batch 81:  Loss:     1.4995 Training Accuracy: 0.828125
    Epoch  1, Sentence Batch 82:  Loss:     1.4796 Training Accuracy: 0.812500
    Epoch  1, Sentence Batch 83:  Loss:     1.5412 Training Accuracy: 0.687500
    Epoch  1, Sentence Batch 84:  Loss:     1.5338 Training Accuracy: 0.734375
    Epoch  1, Sentence Batch 85:  Loss:     1.5469 Training Accuracy: 0.687500
    Epoch  1, Sentence Batch 86:  Loss:     1.4729 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 87:  Loss:     1.4735 Training Accuracy: 0.781250
    Epoch  1, Sentence Batch 88:  Loss:     1.4657 Training Accuracy: 0.781250
    Epoch  1, Sentence Batch 89:  Loss:     1.4183 Training Accuracy: 0.734375
    Epoch  1, Sentence Batch 90:  Loss:     1.3462 Training Accuracy: 0.859375
    Epoch  1, Sentence Batch 91:  Loss:     1.3563 Training Accuracy: 0.843750
    Epoch  1, Sentence Batch 92:  Loss:     1.4393 Training Accuracy: 0.703125
    Epoch  1, Sentence Batch 93:  Loss:     1.3922 Training Accuracy: 0.750000
    Epoch  1, Sentence Batch 94:  Loss:     1.4683 Training Accuracy: 0.640625
    Epoch  1, Sentence Batch 95:  Loss:     1.3968 Training Accuracy: 0.656250
    Epoch  1, Sentence Batch 96:  Loss:     1.4524 Training Accuracy: 0.625000
    Epoch  1, Sentence Batch 97:  Loss:     1.3363 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 98:  Loss:     1.2865 Training Accuracy: 0.812500
    Epoch  1, Sentence Batch 99:  Loss:     1.2783 Training Accuracy: 0.828125
    Epoch  1, Sentence Batch 100:  Validation Accuracy: 0.726079
    Loss:     1.2806 Training Accuracy: 0.796875
    Epoch  1, Sentence Batch 101:  Loss:     1.3274 Training Accuracy: 0.671875
    Epoch  1, Sentence Batch 102:  Loss:     1.2445 Training Accuracy: 0.781250
    Epoch  1, Sentence Batch 103:  Loss:     1.3084 Training Accuracy: 0.703125
    Epoch  1, Sentence Batch 104:  Loss:     1.2434 Training Accuracy: 0.781250
    Epoch  1, Sentence Batch 105:  Loss:     1.1909 Training Accuracy: 0.812500
    Epoch  1, Sentence Batch 106:  Loss:     1.2828 Training Accuracy: 0.671875
    Epoch  1, Sentence Batch 107:  Loss:     1.2235 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 108:  Loss:     1.2106 Training Accuracy: 0.718750
    Epoch  1, Sentence Batch 109:  Loss:     1.1730 Training Accuracy: 0.781250
    Epoch  1, Sentence Batch 110:  Loss:     1.2007 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 111:  Loss:     1.1256 Training Accuracy: 0.812500
    Epoch  1, Sentence Batch 112:  Loss:     1.1375 Training Accuracy: 0.796875
    Epoch  1, Sentence Batch 113:  Loss:     1.1404 Training Accuracy: 0.796875
    Epoch  1, Sentence Batch 114:  Loss:     1.1452 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 115:  Loss:     1.1244 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 116:  Loss:     1.1571 Training Accuracy: 0.750000
    Epoch  1, Sentence Batch 117:  Loss:     1.1274 Training Accuracy: 0.750000
    Epoch  1, Sentence Batch 118:  Loss:     1.1528 Training Accuracy: 0.734375
    Epoch  1, Sentence Batch 119:  Loss:     1.1205 Training Accuracy: 0.796875
    Epoch  1, Sentence Batch 120:  Loss:     1.0859 Training Accuracy: 0.812500
    Epoch  1, Sentence Batch 121:  Loss:     1.1092 Training Accuracy: 0.750000
    Epoch  1, Sentence Batch 122:  Loss:     1.0568 Training Accuracy: 0.781250
    Epoch  1, Sentence Batch 123:  Loss:     1.1002 Training Accuracy: 0.703125
    Epoch  1, Sentence Batch 124:  Loss:     1.0564 Training Accuracy: 0.781250
    Epoch  1, Sentence Batch 125:  Loss:     1.0365 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 126:  Loss:     1.0289 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 127:  Loss:     1.0755 Training Accuracy: 0.687500
    Epoch  1, Sentence Batch 128:  Loss:     0.9615 Training Accuracy: 0.890625
    Epoch  1, Sentence Batch 129:  Loss:     1.0625 Training Accuracy: 0.781250
    Epoch  1, Sentence Batch 130:  Loss:     1.0199 Training Accuracy: 0.796875
    Epoch  1, Sentence Batch 131:  Loss:     1.0260 Training Accuracy: 0.750000
    Epoch  1, Sentence Batch 132:  Loss:     1.0438 Training Accuracy: 0.750000
    Epoch  1, Sentence Batch 133:  Loss:     1.0422 Training Accuracy: 0.781250
    Epoch  1, Sentence Batch 134:  Loss:     0.9815 Training Accuracy: 0.859375
    Epoch  1, Sentence Batch 135:  Loss:     0.9950 Training Accuracy: 0.796875
    Epoch  1, Sentence Batch 136:  Loss:     1.0195 Training Accuracy: 0.734375
    Epoch  1, Sentence Batch 137:  Loss:     0.9823 Training Accuracy: 0.734375
    Epoch  1, Sentence Batch 138:  Loss:     0.9559 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 139:  Loss:     0.9602 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 140:  Loss:     0.9662 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 141:  Loss:     0.9420 Training Accuracy: 0.765625
    Epoch  1, Sentence Batch 142:  Loss:     0.9725 Training Accuracy: 0.703125
    Epoch  1, Sentence Batch 143:  Loss:     0.9736 Training Accuracy: 0.718750
    Epoch  1, Sentence Batch 144:  Loss:     0.9000 Training Accuracy: 0.750000
    Epoch  1, Sentence Batch 145:  Loss:     0.8907 Training Accuracy: 0.796875
    Epoch  1, Sentence Batch 146:  Loss:     0.8956 Training Accuracy: 0.875000
    Epoch  1, Sentence Batch 147:  Loss:     0.8975 Training Accuracy: 0.828125
    Epoch  1, Sentence Batch 148:  Loss:     0.9723 Training Accuracy: 0.656250
    Epoch  1, Sentence Batch 149:  Loss:     0.8905 Training Accuracy: 0.766667
    Epoch  2, Sentence Batch 0:  Loss:     0.8084 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 1:  Loss:     0.8296 Training Accuracy: 0.906250
    Epoch  2, Sentence Batch 2:  Loss:     0.7899 Training Accuracy: 0.890625
    Epoch  2, Sentence Batch 3:  Loss:     0.8347 Training Accuracy: 0.906250
    Epoch  2, Sentence Batch 4:  Loss:     0.8480 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 5:  Loss:     0.8249 Training Accuracy: 0.812500
    Epoch  2, Sentence Batch 6:  Loss:     0.7907 Training Accuracy: 0.890625
    Epoch  2, Sentence Batch 7:  Loss:     0.7921 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 8:  Loss:     0.7933 Training Accuracy: 0.875000
    Epoch  2, Sentence Batch 9:  Loss:     0.7991 Training Accuracy: 0.875000
    Epoch  2, Sentence Batch 10:  Loss:     0.7540 Training Accuracy: 0.937500
    Epoch  2, Sentence Batch 11:  Loss:     0.7683 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 12:  Loss:     0.7674 Training Accuracy: 0.890625
    Epoch  2, Sentence Batch 13:  Loss:     0.7532 Training Accuracy: 0.875000
    Epoch  2, Sentence Batch 14:  Loss:     0.7500 Training Accuracy: 0.906250
    Epoch  2, Sentence Batch 15:  Loss:     0.7933 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 16:  Loss:     0.7498 Training Accuracy: 0.906250
    Epoch  2, Sentence Batch 17:  Loss:     0.7682 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 18:  Loss:     0.7272 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 19:  Loss:     0.7091 Training Accuracy: 0.921875
    Epoch  2, Sentence Batch 20:  Loss:     0.7628 Training Accuracy: 0.875000
    Epoch  2, Sentence Batch 21:  Loss:     0.6851 Training Accuracy: 0.906250
    Epoch  2, Sentence Batch 22:  Loss:     0.7304 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 23:  Loss:     0.7831 Training Accuracy: 0.750000
    Epoch  2, Sentence Batch 24:  Loss:     0.7339 Training Accuracy: 0.828125
    Epoch  2, Sentence Batch 25:  Loss:     0.7847 Training Accuracy: 0.765625
    Epoch  2, Sentence Batch 26:  Loss:     0.6665 Training Accuracy: 0.906250
    Epoch  2, Sentence Batch 27:  Loss:     0.7393 Training Accuracy: 0.828125
    Epoch  2, Sentence Batch 28:  Loss:     0.7236 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 29:  Loss:     0.7024 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 30:  Loss:     0.6668 Training Accuracy: 0.921875
    Epoch  2, Sentence Batch 31:  Loss:     0.7081 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 32:  Loss:     0.7357 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 33:  Loss:     0.7235 Training Accuracy: 0.781250
    Epoch  2, Sentence Batch 34:  Loss:     0.7246 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 35:  Loss:     0.6814 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 36:  Loss:     0.7091 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 37:  Loss:     0.7179 Training Accuracy: 0.812500
    Epoch  2, Sentence Batch 38:  Loss:     0.7571 Training Accuracy: 0.796875
    Epoch  2, Sentence Batch 39:  Loss:     0.6500 Training Accuracy: 0.890625
    Epoch  2, Sentence Batch 40:  Loss:     0.6397 Training Accuracy: 0.937500
    Epoch  2, Sentence Batch 41:  Loss:     0.6774 Training Accuracy: 0.890625
    Epoch  2, Sentence Batch 42:  Loss:     0.7050 Training Accuracy: 0.828125
    Epoch  2, Sentence Batch 43:  Loss:     0.6779 Training Accuracy: 0.812500
    Epoch  2, Sentence Batch 44:  Loss:     0.6905 Training Accuracy: 0.812500
    Epoch  2, Sentence Batch 45:  Loss:     0.6172 Training Accuracy: 0.875000
    Epoch  2, Sentence Batch 46:  Loss:     0.6673 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 47:  Loss:     0.6601 Training Accuracy: 0.812500
    Epoch  2, Sentence Batch 48:  Loss:     0.6660 Training Accuracy: 0.812500
    Epoch  2, Sentence Batch 49:  Loss:     0.6343 Training Accuracy: 0.875000
    Epoch  2, Sentence Batch 50:  Validation Accuracy: 0.743902
    Loss:     0.6352 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 51:  Loss:     0.6814 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 52:  Loss:     0.6601 Training Accuracy: 0.875000
    Epoch  2, Sentence Batch 53:  Loss:     0.6518 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 54:  Loss:     0.6446 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 55:  Loss:     0.6456 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 56:  Loss:     0.7295 Training Accuracy: 0.734375
    Epoch  2, Sentence Batch 57:  Loss:     0.6621 Training Accuracy: 0.781250
    Epoch  2, Sentence Batch 58:  Loss:     0.6425 Training Accuracy: 0.812500
    Epoch  2, Sentence Batch 59:  Loss:     0.5816 Training Accuracy: 0.937500
    Epoch  2, Sentence Batch 60:  Loss:     0.6531 Training Accuracy: 0.796875
    Epoch  2, Sentence Batch 61:  Loss:     0.6768 Training Accuracy: 0.765625
    Epoch  2, Sentence Batch 62:  Loss:     0.6450 Training Accuracy: 0.796875
    Epoch  2, Sentence Batch 63:  Loss:     0.5778 Training Accuracy: 0.906250
    Epoch  2, Sentence Batch 64:  Loss:     0.6457 Training Accuracy: 0.812500
    Epoch  2, Sentence Batch 65:  Loss:     0.5859 Training Accuracy: 0.890625
    Epoch  2, Sentence Batch 66:  Loss:     0.6184 Training Accuracy: 0.828125
    Epoch  2, Sentence Batch 67:  Loss:     0.6303 Training Accuracy: 0.796875
    Epoch  2, Sentence Batch 68:  Loss:     0.6600 Training Accuracy: 0.812500
    Epoch  2, Sentence Batch 69:  Loss:     0.6766 Training Accuracy: 0.781250
    Epoch  2, Sentence Batch 70:  Loss:     0.6276 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 71:  Loss:     0.6326 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 72:  Loss:     0.6422 Training Accuracy: 0.750000
    Epoch  2, Sentence Batch 73:  Loss:     0.6354 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 74:  Loss:     0.5672 Training Accuracy: 0.937500
    Epoch  2, Sentence Batch 75:  Loss:     0.5875 Training Accuracy: 0.875000
    Epoch  2, Sentence Batch 76:  Loss:     0.5774 Training Accuracy: 0.812500
    Epoch  2, Sentence Batch 77:  Loss:     0.6789 Training Accuracy: 0.734375
    Epoch  2, Sentence Batch 78:  Loss:     0.5858 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 79:  Loss:     0.6162 Training Accuracy: 0.781250
    Epoch  2, Sentence Batch 80:  Loss:     0.6117 Training Accuracy: 0.890625
    Epoch  2, Sentence Batch 81:  Loss:     0.6156 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 82:  Loss:     0.6214 Training Accuracy: 0.828125
    Epoch  2, Sentence Batch 83:  Loss:     0.5873 Training Accuracy: 0.921875
    Epoch  2, Sentence Batch 84:  Loss:     0.6242 Training Accuracy: 0.796875
    Epoch  2, Sentence Batch 85:  Loss:     0.5878 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 86:  Loss:     0.5761 Training Accuracy: 0.812500
    Epoch  2, Sentence Batch 87:  Loss:     0.5212 Training Accuracy: 0.906250
    Epoch  2, Sentence Batch 88:  Loss:     0.5668 Training Accuracy: 0.890625
    Epoch  2, Sentence Batch 89:  Loss:     0.6319 Training Accuracy: 0.812500
    Epoch  2, Sentence Batch 90:  Loss:     0.5843 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 91:  Loss:     0.5709 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 92:  Loss:     0.5986 Training Accuracy: 0.765625
    Epoch  2, Sentence Batch 93:  Loss:     0.5729 Training Accuracy: 0.875000
    Epoch  2, Sentence Batch 94:  Loss:     0.5836 Training Accuracy: 0.875000
    Epoch  2, Sentence Batch 95:  Loss:     0.5719 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 96:  Loss:     0.5699 Training Accuracy: 0.796875
    Epoch  2, Sentence Batch 97:  Loss:     0.5868 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 98:  Loss:     0.5249 Training Accuracy: 0.906250
    Epoch  2, Sentence Batch 99:  Loss:     0.5378 Training Accuracy: 0.921875
    Epoch  2, Sentence Batch 100:  Loss:     0.5800 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 101:  Loss:     0.5793 Training Accuracy: 0.765625
    Epoch  2, Sentence Batch 102:  Loss:     0.5178 Training Accuracy: 0.921875
    Epoch  2, Sentence Batch 103:  Loss:     0.5907 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 104:  Loss:     0.5305 Training Accuracy: 0.890625
    Epoch  2, Sentence Batch 105:  Loss:     0.6078 Training Accuracy: 0.796875
    Epoch  2, Sentence Batch 106:  Loss:     0.5666 Training Accuracy: 0.828125
    Epoch  2, Sentence Batch 107:  Loss:     0.5408 Training Accuracy: 0.890625
    Epoch  2, Sentence Batch 108:  Loss:     0.5447 Training Accuracy: 0.937500
    Epoch  2, Sentence Batch 109:  Loss:     0.5946 Training Accuracy: 0.781250
    Epoch  2, Sentence Batch 110:  Loss:     0.5296 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 111:  Loss:     0.5386 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 112:  Loss:     0.5347 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 113:  Loss:     0.5419 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 114:  Loss:     0.5664 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 115:  Loss:     0.5607 Training Accuracy: 0.796875
    Epoch  2, Sentence Batch 116:  Loss:     0.6088 Training Accuracy: 0.781250
    Epoch  2, Sentence Batch 117:  Loss:     0.5559 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 118:  Loss:     0.5664 Training Accuracy: 0.828125
    Epoch  2, Sentence Batch 119:  Loss:     0.4793 Training Accuracy: 0.953125
    Epoch  2, Sentence Batch 120:  Loss:     0.6107 Training Accuracy: 0.734375
    Epoch  2, Sentence Batch 121:  Loss:     0.5510 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 122:  Loss:     0.5533 Training Accuracy: 0.828125
    Epoch  2, Sentence Batch 123:  Loss:     0.5513 Training Accuracy: 0.875000
    Epoch  2, Sentence Batch 124:  Loss:     0.5546 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 125:  Loss:     0.5303 Training Accuracy: 0.750000
    Epoch  2, Sentence Batch 126:  Loss:     0.5596 Training Accuracy: 0.812500
    Epoch  2, Sentence Batch 127:  Loss:     0.5271 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 128:  Loss:     0.5689 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 129:  Loss:     0.5059 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 130:  Loss:     0.5288 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 131:  Loss:     0.5830 Training Accuracy: 0.765625
    Epoch  2, Sentence Batch 132:  Loss:     0.5842 Training Accuracy: 0.781250
    Epoch  2, Sentence Batch 133:  Loss:     0.5545 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 134:  Loss:     0.5306 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 135:  Loss:     0.5067 Training Accuracy: 0.875000
    Epoch  2, Sentence Batch 136:  Loss:     0.5414 Training Accuracy: 0.859375
    Epoch  2, Sentence Batch 137:  Loss:     0.4731 Training Accuracy: 0.843750
    Epoch  2, Sentence Batch 138:  Loss:     0.5768 Training Accuracy: 0.765625
    Epoch  2, Sentence Batch 139:  Loss:     0.5338 Training Accuracy: 0.875000
    Epoch  2, Sentence Batch 140:  Loss:     0.5040 Training Accuracy: 0.812500
    Epoch  2, Sentence Batch 141:  Loss:     0.5050 Training Accuracy: 0.828125
    Epoch  2, Sentence Batch 142:  Loss:     0.4805 Training Accuracy: 0.890625
    Epoch  2, Sentence Batch 143:  Loss:     0.5319 Training Accuracy: 0.828125
    Epoch  2, Sentence Batch 144:  Loss:     0.5612 Training Accuracy: 0.703125
    Epoch  2, Sentence Batch 145:  Loss:     0.4749 Training Accuracy: 0.890625
    Epoch  2, Sentence Batch 146:  Loss:     0.5504 Training Accuracy: 0.812500
    Epoch  2, Sentence Batch 147:  Loss:     0.5066 Training Accuracy: 0.890625
    Epoch  2, Sentence Batch 148:  Loss:     0.4904 Training Accuracy: 0.890625
    Epoch  2, Sentence Batch 149:  Loss:     0.6013 Training Accuracy: 0.783333
    Epoch  3, Sentence Batch 0:  Validation Accuracy: 0.751407
    Loss:     0.4127 Training Accuracy: 0.953125
    Epoch  3, Sentence Batch 1:  Loss:     0.4934 Training Accuracy: 0.843750
    Epoch  3, Sentence Batch 2:  Loss:     0.4635 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 3:  Loss:     0.4390 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 4:  Loss:     0.4678 Training Accuracy: 0.875000
    Epoch  3, Sentence Batch 5:  Loss:     0.4912 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 6:  Loss:     0.4660 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 7:  Loss:     0.4378 Training Accuracy: 0.937500
    Epoch  3, Sentence Batch 8:  Loss:     0.4132 Training Accuracy: 0.937500
    Epoch  3, Sentence Batch 9:  Loss:     0.4565 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 10:  Loss:     0.5234 Training Accuracy: 0.843750
    Epoch  3, Sentence Batch 11:  Loss:     0.5176 Training Accuracy: 0.796875
    Epoch  3, Sentence Batch 12:  Loss:     0.4911 Training Accuracy: 0.875000
    Epoch  3, Sentence Batch 13:  Loss:     0.4615 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 14:  Loss:     0.4360 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 15:  Loss:     0.4624 Training Accuracy: 0.843750
    Epoch  3, Sentence Batch 16:  Loss:     0.4871 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 17:  Loss:     0.4268 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 18:  Loss:     0.4387 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 19:  Loss:     0.4562 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 20:  Loss:     0.5094 Training Accuracy: 0.843750
    Epoch  3, Sentence Batch 21:  Loss:     0.4075 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 22:  Loss:     0.4404 Training Accuracy: 0.875000
    Epoch  3, Sentence Batch 23:  Loss:     0.4637 Training Accuracy: 0.875000
    Epoch  3, Sentence Batch 24:  Loss:     0.4200 Training Accuracy: 0.953125
    Epoch  3, Sentence Batch 25:  Loss:     0.5031 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 26:  Loss:     0.4846 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 27:  Loss:     0.4430 Training Accuracy: 0.937500
    Epoch  3, Sentence Batch 28:  Loss:     0.4463 Training Accuracy: 0.875000
    Epoch  3, Sentence Batch 29:  Loss:     0.4951 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 30:  Loss:     0.4750 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 31:  Loss:     0.4684 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 32:  Loss:     0.4363 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 33:  Loss:     0.4254 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 34:  Loss:     0.4079 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 35:  Loss:     0.4102 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 36:  Loss:     0.4149 Training Accuracy: 0.937500
    Epoch  3, Sentence Batch 37:  Loss:     0.4046 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 38:  Loss:     0.4379 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 39:  Loss:     0.4131 Training Accuracy: 0.937500
    Epoch  3, Sentence Batch 40:  Loss:     0.4347 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 41:  Loss:     0.4087 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 42:  Loss:     0.4420 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 43:  Loss:     0.5038 Training Accuracy: 0.796875
    Epoch  3, Sentence Batch 44:  Loss:     0.4295 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 45:  Loss:     0.4486 Training Accuracy: 0.843750
    Epoch  3, Sentence Batch 46:  Loss:     0.3731 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 47:  Loss:     0.4126 Training Accuracy: 0.937500
    Epoch  3, Sentence Batch 48:  Loss:     0.3944 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 49:  Loss:     0.4226 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 50:  Loss:     0.4170 Training Accuracy: 0.875000
    Epoch  3, Sentence Batch 51:  Loss:     0.4274 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 52:  Loss:     0.5149 Training Accuracy: 0.781250
    Epoch  3, Sentence Batch 53:  Loss:     0.3887 Training Accuracy: 0.937500
    Epoch  3, Sentence Batch 54:  Loss:     0.4448 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 55:  Loss:     0.4406 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 56:  Loss:     0.4300 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 57:  Loss:     0.3754 Training Accuracy: 0.953125
    Epoch  3, Sentence Batch 58:  Loss:     0.4776 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 59:  Loss:     0.3833 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 60:  Loss:     0.4000 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 61:  Loss:     0.3937 Training Accuracy: 0.875000
    Epoch  3, Sentence Batch 62:  Loss:     0.4409 Training Accuracy: 0.875000
    Epoch  3, Sentence Batch 63:  Loss:     0.3802 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 64:  Loss:     0.4382 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 65:  Loss:     0.4146 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 66:  Loss:     0.4507 Training Accuracy: 0.843750
    Epoch  3, Sentence Batch 67:  Loss:     0.3998 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 68:  Loss:     0.4304 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 69:  Loss:     0.4223 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 70:  Loss:     0.4275 Training Accuracy: 0.843750
    Epoch  3, Sentence Batch 71:  Loss:     0.4488 Training Accuracy: 0.843750
    Epoch  3, Sentence Batch 72:  Loss:     0.5074 Training Accuracy: 0.812500
    Epoch  3, Sentence Batch 73:  Loss:     0.4537 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 74:  Loss:     0.4543 Training Accuracy: 0.828125
    Epoch  3, Sentence Batch 75:  Loss:     0.4532 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 76:  Loss:     0.4213 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 77:  Loss:     0.4542 Training Accuracy: 0.828125
    Epoch  3, Sentence Batch 78:  Loss:     0.4300 Training Accuracy: 0.875000
    Epoch  3, Sentence Batch 79:  Loss:     0.3925 Training Accuracy: 0.875000
    Epoch  3, Sentence Batch 80:  Loss:     0.4151 Training Accuracy: 0.937500
    Epoch  3, Sentence Batch 81:  Loss:     0.3689 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 82:  Loss:     0.4600 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 83:  Loss:     0.4206 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 84:  Loss:     0.4278 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 85:  Loss:     0.4490 Training Accuracy: 0.812500
    Epoch  3, Sentence Batch 86:  Loss:     0.4737 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 87:  Loss:     0.4307 Training Accuracy: 0.843750
    Epoch  3, Sentence Batch 88:  Loss:     0.4144 Training Accuracy: 0.875000
    Epoch  3, Sentence Batch 89:  Loss:     0.4045 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 90:  Loss:     0.4090 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 91:  Loss:     0.4788 Training Accuracy: 0.828125
    Epoch  3, Sentence Batch 92:  Loss:     0.4096 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 93:  Loss:     0.3635 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 94:  Loss:     0.3739 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 95:  Loss:     0.4072 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 96:  Loss:     0.3828 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 97:  Loss:     0.4309 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 98:  Loss:     0.4306 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 99:  Loss:     0.3806 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 100:  Validation Accuracy: 0.773921
    Loss:     0.3701 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 101:  Loss:     0.3884 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 102:  Loss:     0.4105 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 103:  Loss:     0.5048 Training Accuracy: 0.750000
    Epoch  3, Sentence Batch 104:  Loss:     0.4666 Training Accuracy: 0.828125
    Epoch  3, Sentence Batch 105:  Loss:     0.4412 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 106:  Loss:     0.4575 Training Accuracy: 0.781250
    Epoch  3, Sentence Batch 107:  Loss:     0.4364 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 108:  Loss:     0.4182 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 109:  Loss:     0.4743 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 110:  Loss:     0.3862 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 111:  Loss:     0.5681 Training Accuracy: 0.750000
    Epoch  3, Sentence Batch 112:  Loss:     0.4590 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 113:  Loss:     0.4289 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 114:  Loss:     0.4575 Training Accuracy: 0.796875
    Epoch  3, Sentence Batch 115:  Loss:     0.3338 Training Accuracy: 0.937500
    Epoch  3, Sentence Batch 116:  Loss:     0.4225 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 117:  Loss:     0.4646 Training Accuracy: 0.843750
    Epoch  3, Sentence Batch 118:  Loss:     0.4029 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 119:  Loss:     0.3849 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 120:  Loss:     0.4238 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 121:  Loss:     0.3963 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 122:  Loss:     0.3667 Training Accuracy: 0.937500
    Epoch  3, Sentence Batch 123:  Loss:     0.4648 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 124:  Loss:     0.4334 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 125:  Loss:     0.4461 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 126:  Loss:     0.3722 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 127:  Loss:     0.4842 Training Accuracy: 0.875000
    Epoch  3, Sentence Batch 128:  Loss:     0.4311 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 129:  Loss:     0.4237 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 130:  Loss:     0.4285 Training Accuracy: 0.843750
    Epoch  3, Sentence Batch 131:  Loss:     0.3770 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 132:  Loss:     0.4585 Training Accuracy: 0.843750
    Epoch  3, Sentence Batch 133:  Loss:     0.3935 Training Accuracy: 0.875000
    Epoch  3, Sentence Batch 134:  Loss:     0.3630 Training Accuracy: 0.937500
    Epoch  3, Sentence Batch 135:  Loss:     0.4279 Training Accuracy: 0.828125
    Epoch  3, Sentence Batch 136:  Loss:     0.3607 Training Accuracy: 0.937500
    Epoch  3, Sentence Batch 137:  Loss:     0.4164 Training Accuracy: 0.828125
    Epoch  3, Sentence Batch 138:  Loss:     0.3612 Training Accuracy: 0.937500
    Epoch  3, Sentence Batch 139:  Loss:     0.4483 Training Accuracy: 0.843750
    Epoch  3, Sentence Batch 140:  Loss:     0.3867 Training Accuracy: 0.890625
    Epoch  3, Sentence Batch 141:  Loss:     0.3404 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 142:  Loss:     0.3839 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 143:  Loss:     0.3981 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 144:  Loss:     0.4365 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 145:  Loss:     0.4134 Training Accuracy: 0.875000
    Epoch  3, Sentence Batch 146:  Loss:     0.4385 Training Accuracy: 0.859375
    Epoch  3, Sentence Batch 147:  Loss:     0.3770 Training Accuracy: 0.921875
    Epoch  3, Sentence Batch 148:  Loss:     0.3699 Training Accuracy: 0.906250
    Epoch  3, Sentence Batch 149:  Loss:     0.3758 Training Accuracy: 0.933333
    Epoch  4, Sentence Batch 0:  Loss:     0.3661 Training Accuracy: 0.968750
    Epoch  4, Sentence Batch 1:  Loss:     0.3558 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 2:  Loss:     0.3276 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 3:  Loss:     0.3314 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 4:  Loss:     0.3122 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 5:  Loss:     0.3362 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 6:  Loss:     0.3899 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 7:  Loss:     0.3631 Training Accuracy: 0.859375
    Epoch  4, Sentence Batch 8:  Loss:     0.4114 Training Accuracy: 0.843750
    Epoch  4, Sentence Batch 9:  Loss:     0.3280 Training Accuracy: 0.968750
    Epoch  4, Sentence Batch 10:  Loss:     0.4389 Training Accuracy: 0.828125
    Epoch  4, Sentence Batch 11:  Loss:     0.3341 Training Accuracy: 0.968750
    Epoch  4, Sentence Batch 12:  Loss:     0.3384 Training Accuracy: 0.968750
    Epoch  4, Sentence Batch 13:  Loss:     0.3677 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 14:  Loss:     0.3113 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 15:  Loss:     0.3519 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 16:  Loss:     0.3379 Training Accuracy: 0.890625
    Epoch  4, Sentence Batch 17:  Loss:     0.3439 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 18:  Loss:     0.3297 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 19:  Loss:     0.3343 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 20:  Loss:     0.3361 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 21:  Loss:     0.2853 Training Accuracy: 0.984375
    Epoch  4, Sentence Batch 22:  Loss:     0.2549 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 23:  Loss:     0.3215 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 24:  Loss:     0.3878 Training Accuracy: 0.875000
    Epoch  4, Sentence Batch 25:  Loss:     0.3392 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 26:  Loss:     0.3353 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 27:  Loss:     0.3419 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 28:  Loss:     0.3917 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 29:  Loss:     0.3848 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 30:  Loss:     0.3977 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 31:  Loss:     0.3629 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 32:  Loss:     0.3772 Training Accuracy: 0.843750
    Epoch  4, Sentence Batch 33:  Loss:     0.3391 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 34:  Loss:     0.3501 Training Accuracy: 0.890625
    Epoch  4, Sentence Batch 35:  Loss:     0.3779 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 36:  Loss:     0.3025 Training Accuracy: 0.968750
    Epoch  4, Sentence Batch 37:  Loss:     0.3347 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 38:  Loss:     0.2779 Training Accuracy: 0.968750
    Epoch  4, Sentence Batch 39:  Loss:     0.3974 Training Accuracy: 0.875000
    Epoch  4, Sentence Batch 40:  Loss:     0.3514 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 41:  Loss:     0.3249 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 42:  Loss:     0.3177 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 43:  Loss:     0.3843 Training Accuracy: 0.875000
    Epoch  4, Sentence Batch 44:  Loss:     0.3433 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 45:  Loss:     0.4110 Training Accuracy: 0.890625
    Epoch  4, Sentence Batch 46:  Loss:     0.2891 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 47:  Loss:     0.3247 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 48:  Loss:     0.2902 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 49:  Loss:     0.2843 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 50:  Validation Accuracy: 0.776735
    Loss:     0.3043 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 51:  Loss:     0.2554 Training Accuracy: 0.968750
    Epoch  4, Sentence Batch 52:  Loss:     0.3448 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 53:  Loss:     0.3777 Training Accuracy: 0.843750
    Epoch  4, Sentence Batch 54:  Loss:     0.3071 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 55:  Loss:     0.2962 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 56:  Loss:     0.3539 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 57:  Loss:     0.3212 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 58:  Loss:     0.3677 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 59:  Loss:     0.3614 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 60:  Loss:     0.2993 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 61:  Loss:     0.3067 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 62:  Loss:     0.3167 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 63:  Loss:     0.3673 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 64:  Loss:     0.3419 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 65:  Loss:     0.3267 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 66:  Loss:     0.3089 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 67:  Loss:     0.3094 Training Accuracy: 0.968750
    Epoch  4, Sentence Batch 68:  Loss:     0.3305 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 69:  Loss:     0.4072 Training Accuracy: 0.875000
    Epoch  4, Sentence Batch 70:  Loss:     0.3302 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 71:  Loss:     0.2883 Training Accuracy: 0.984375
    Epoch  4, Sentence Batch 72:  Loss:     0.3686 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 73:  Loss:     0.3462 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 74:  Loss:     0.2639 Training Accuracy: 0.984375
    Epoch  4, Sentence Batch 75:  Loss:     0.3559 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 76:  Loss:     0.3515 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 77:  Loss:     0.2987 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 78:  Loss:     0.3671 Training Accuracy: 0.859375
    Epoch  4, Sentence Batch 79:  Loss:     0.2608 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 80:  Loss:     0.3005 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 81:  Loss:     0.2986 Training Accuracy: 0.968750
    Epoch  4, Sentence Batch 82:  Loss:     0.3095 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 83:  Loss:     0.3026 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 84:  Loss:     0.2921 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 85:  Loss:     0.3234 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 86:  Loss:     0.2920 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 87:  Loss:     0.2999 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 88:  Loss:     0.3359 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 89:  Loss:     0.3634 Training Accuracy: 0.890625
    Epoch  4, Sentence Batch 90:  Loss:     0.2775 Training Accuracy: 0.984375
    Epoch  4, Sentence Batch 91:  Loss:     0.3838 Training Accuracy: 0.890625
    Epoch  4, Sentence Batch 92:  Loss:     0.3147 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 93:  Loss:     0.3035 Training Accuracy: 0.968750
    Epoch  4, Sentence Batch 94:  Loss:     0.3882 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 95:  Loss:     0.3655 Training Accuracy: 0.859375
    Epoch  4, Sentence Batch 96:  Loss:     0.2618 Training Accuracy: 0.968750
    Epoch  4, Sentence Batch 97:  Loss:     0.3612 Training Accuracy: 0.890625
    Epoch  4, Sentence Batch 98:  Loss:     0.3835 Training Accuracy: 0.859375
    Epoch  4, Sentence Batch 99:  Loss:     0.3240 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 100:  Loss:     0.3154 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 101:  Loss:     0.2769 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 102:  Loss:     0.2774 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 103:  Loss:     0.3277 Training Accuracy: 0.890625
    Epoch  4, Sentence Batch 104:  Loss:     0.3463 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 105:  Loss:     0.2854 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 106:  Loss:     0.2830 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 107:  Loss:     0.2704 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 108:  Loss:     0.3114 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 109:  Loss:     0.2978 Training Accuracy: 0.968750
    Epoch  4, Sentence Batch 110:  Loss:     0.3992 Training Accuracy: 0.859375
    Epoch  4, Sentence Batch 111:  Loss:     0.3788 Training Accuracy: 0.812500
    Epoch  4, Sentence Batch 112:  Loss:     0.3808 Training Accuracy: 0.875000
    Epoch  4, Sentence Batch 113:  Loss:     0.3221 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 114:  Loss:     0.3809 Training Accuracy: 0.890625
    Epoch  4, Sentence Batch 115:  Loss:     0.3245 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 116:  Loss:     0.3524 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 117:  Loss:     0.3284 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 118:  Loss:     0.3297 Training Accuracy: 0.890625
    Epoch  4, Sentence Batch 119:  Loss:     0.3461 Training Accuracy: 0.890625
    Epoch  4, Sentence Batch 120:  Loss:     0.2745 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 121:  Loss:     0.3036 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 122:  Loss:     0.3140 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 123:  Loss:     0.2881 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 124:  Loss:     0.3370 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 125:  Loss:     0.3225 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 126:  Loss:     0.2812 Training Accuracy: 0.968750
    Epoch  4, Sentence Batch 127:  Loss:     0.3077 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 128:  Loss:     0.3522 Training Accuracy: 0.890625
    Epoch  4, Sentence Batch 129:  Loss:     0.3541 Training Accuracy: 0.937500
    Epoch  4, Sentence Batch 130:  Loss:     0.3003 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 131:  Loss:     0.3935 Training Accuracy: 0.875000
    Epoch  4, Sentence Batch 132:  Loss:     0.2790 Training Accuracy: 0.968750
    Epoch  4, Sentence Batch 133:  Loss:     0.3664 Training Accuracy: 0.875000
    Epoch  4, Sentence Batch 134:  Loss:     0.3813 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 135:  Loss:     0.2997 Training Accuracy: 0.906250
    Epoch  4, Sentence Batch 136:  Loss:     0.4205 Training Accuracy: 0.843750
    Epoch  4, Sentence Batch 137:  Loss:     0.3794 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 138:  Loss:     0.3249 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 139:  Loss:     0.5023 Training Accuracy: 0.765625
    Epoch  4, Sentence Batch 140:  Loss:     0.3076 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 141:  Loss:     0.3988 Training Accuracy: 0.843750
    Epoch  4, Sentence Batch 142:  Loss:     0.3458 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 143:  Loss:     0.3244 Training Accuracy: 0.953125
    Epoch  4, Sentence Batch 144:  Loss:     0.2918 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 145:  Loss:     0.3072 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 146:  Loss:     0.3986 Training Accuracy: 0.859375
    Epoch  4, Sentence Batch 147:  Loss:     0.3576 Training Accuracy: 0.859375
    Epoch  4, Sentence Batch 148:  Loss:     0.2988 Training Accuracy: 0.921875
    Epoch  4, Sentence Batch 149:  Loss:     0.2913 Training Accuracy: 0.933333
    Epoch  5, Sentence Batch 0:  Validation Accuracy: 0.777674
    Loss:     0.2747 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 1:  Loss:     0.2320 Training Accuracy: 0.984375
    Epoch  5, Sentence Batch 2:  Loss:     0.2670 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 3:  Loss:     0.3048 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 4:  Loss:     0.2642 Training Accuracy: 0.984375
    Epoch  5, Sentence Batch 5:  Loss:     0.2905 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 6:  Loss:     0.2934 Training Accuracy: 0.875000
    Epoch  5, Sentence Batch 7:  Loss:     0.3127 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 8:  Loss:     0.2228 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 9:  Loss:     0.2684 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 10:  Loss:     0.2294 Training Accuracy: 0.984375
    Epoch  5, Sentence Batch 11:  Loss:     0.2639 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 12:  Loss:     0.2777 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 13:  Loss:     0.2857 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 14:  Loss:     0.2592 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 15:  Loss:     0.2550 Training Accuracy: 0.984375
    Epoch  5, Sentence Batch 16:  Loss:     0.3297 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 17:  Loss:     0.3111 Training Accuracy: 0.921875
    Epoch  5, Sentence Batch 18:  Loss:     0.2821 Training Accuracy: 0.906250
    Epoch  5, Sentence Batch 19:  Loss:     0.2305 Training Accuracy: 0.984375
    Epoch  5, Sentence Batch 20:  Loss:     0.2260 Training Accuracy: 1.000000
    Epoch  5, Sentence Batch 21:  Loss:     0.2984 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 22:  Loss:     0.2860 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 23:  Loss:     0.2412 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 24:  Loss:     0.2633 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 25:  Loss:     0.2612 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 26:  Loss:     0.2995 Training Accuracy: 0.906250
    Epoch  5, Sentence Batch 27:  Loss:     0.2825 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 28:  Loss:     0.2634 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 29:  Loss:     0.3078 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 30:  Loss:     0.3047 Training Accuracy: 0.906250
    Epoch  5, Sentence Batch 31:  Loss:     0.3039 Training Accuracy: 0.906250
    Epoch  5, Sentence Batch 32:  Loss:     0.2883 Training Accuracy: 0.984375
    Epoch  5, Sentence Batch 33:  Loss:     0.2471 Training Accuracy: 1.000000
    Epoch  5, Sentence Batch 34:  Loss:     0.2673 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 35:  Loss:     0.2431 Training Accuracy: 0.984375
    Epoch  5, Sentence Batch 36:  Loss:     0.3628 Training Accuracy: 0.906250
    Epoch  5, Sentence Batch 37:  Loss:     0.2512 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 38:  Loss:     0.2660 Training Accuracy: 0.921875
    Epoch  5, Sentence Batch 39:  Loss:     0.2833 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 40:  Loss:     0.2402 Training Accuracy: 0.984375
    Epoch  5, Sentence Batch 41:  Loss:     0.1794 Training Accuracy: 1.000000
    Epoch  5, Sentence Batch 42:  Loss:     0.2420 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 43:  Loss:     0.2343 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 44:  Loss:     0.2258 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 45:  Loss:     0.2570 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 46:  Loss:     0.2578 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 47:  Loss:     0.2606 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 48:  Loss:     0.2446 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 49:  Loss:     0.2517 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 50:  Loss:     0.2726 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 51:  Loss:     0.3046 Training Accuracy: 0.921875
    Epoch  5, Sentence Batch 52:  Loss:     0.2156 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 53:  Loss:     0.2429 Training Accuracy: 0.984375
    Epoch  5, Sentence Batch 54:  Loss:     0.2324 Training Accuracy: 1.000000
    Epoch  5, Sentence Batch 55:  Loss:     0.2309 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 56:  Loss:     0.2204 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 57:  Loss:     0.2404 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 58:  Loss:     0.2492 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 59:  Loss:     0.2690 Training Accuracy: 0.921875
    Epoch  5, Sentence Batch 60:  Loss:     0.2867 Training Accuracy: 0.921875
    Epoch  5, Sentence Batch 61:  Loss:     0.2403 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 62:  Loss:     0.2568 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 63:  Loss:     0.2461 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 64:  Loss:     0.2666 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 65:  Loss:     0.2752 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 66:  Loss:     0.2318 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 67:  Loss:     0.2667 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 68:  Loss:     0.2457 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 69:  Loss:     0.2918 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 70:  Loss:     0.2335 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 71:  Loss:     0.2635 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 72:  Loss:     0.3067 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 73:  Loss:     0.2401 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 74:  Loss:     0.2544 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 75:  Loss:     0.2590 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 76:  Loss:     0.2411 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 77:  Loss:     0.2091 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 78:  Loss:     0.2576 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 79:  Loss:     0.2501 Training Accuracy: 0.984375
    Epoch  5, Sentence Batch 80:  Loss:     0.2266 Training Accuracy: 1.000000
    Epoch  5, Sentence Batch 81:  Loss:     0.2477 Training Accuracy: 0.906250
    Epoch  5, Sentence Batch 82:  Loss:     0.3007 Training Accuracy: 0.921875
    Epoch  5, Sentence Batch 83:  Loss:     0.3147 Training Accuracy: 0.906250
    Epoch  5, Sentence Batch 84:  Loss:     0.2365 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 85:  Loss:     0.2392 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 86:  Loss:     0.2955 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 87:  Loss:     0.2620 Training Accuracy: 0.921875
    Epoch  5, Sentence Batch 88:  Loss:     0.2660 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 89:  Loss:     0.2008 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 90:  Loss:     0.2649 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 91:  Loss:     0.2198 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 92:  Loss:     0.2533 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 93:  Loss:     0.3194 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 94:  Loss:     0.2593 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 95:  Loss:     0.2681 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 96:  Loss:     0.3021 Training Accuracy: 0.890625
    Epoch  5, Sentence Batch 97:  Loss:     0.2144 Training Accuracy: 0.984375
    Epoch  5, Sentence Batch 98:  Loss:     0.2721 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 99:  Loss:     0.3332 Training Accuracy: 0.890625
    Epoch  5, Sentence Batch 100:  Validation Accuracy: 0.790807
    Loss:     0.2465 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 101:  Loss:     0.2601 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 102:  Loss:     0.3433 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 103:  Loss:     0.2475 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 104:  Loss:     0.2763 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 105:  Loss:     0.2187 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 106:  Loss:     0.2891 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 107:  Loss:     0.2587 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 108:  Loss:     0.2186 Training Accuracy: 0.984375
    Epoch  5, Sentence Batch 109:  Loss:     0.2656 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 110:  Loss:     0.2698 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 111:  Loss:     0.2359 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 112:  Loss:     0.2864 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 113:  Loss:     0.3144 Training Accuracy: 0.906250
    Epoch  5, Sentence Batch 114:  Loss:     0.2366 Training Accuracy: 0.984375
    Epoch  5, Sentence Batch 115:  Loss:     0.3062 Training Accuracy: 0.859375
    Epoch  5, Sentence Batch 116:  Loss:     0.2664 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 117:  Loss:     0.2488 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 118:  Loss:     0.3111 Training Accuracy: 0.875000
    Epoch  5, Sentence Batch 119:  Loss:     0.2113 Training Accuracy: 0.984375
    Epoch  5, Sentence Batch 120:  Loss:     0.2833 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 121:  Loss:     0.2550 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 122:  Loss:     0.2251 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 123:  Loss:     0.2454 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 124:  Loss:     0.2926 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 125:  Loss:     0.2646 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 126:  Loss:     0.2157 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 127:  Loss:     0.3061 Training Accuracy: 0.890625
    Epoch  5, Sentence Batch 128:  Loss:     0.2473 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 129:  Loss:     0.2883 Training Accuracy: 0.906250
    Epoch  5, Sentence Batch 130:  Loss:     0.2370 Training Accuracy: 0.984375
    Epoch  5, Sentence Batch 131:  Loss:     0.2540 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 132:  Loss:     0.2784 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 133:  Loss:     0.2828 Training Accuracy: 0.921875
    Epoch  5, Sentence Batch 134:  Loss:     0.2570 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 135:  Loss:     0.2782 Training Accuracy: 0.921875
    Epoch  5, Sentence Batch 136:  Loss:     0.2638 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 137:  Loss:     0.2262 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 138:  Loss:     0.2769 Training Accuracy: 0.921875
    Epoch  5, Sentence Batch 139:  Loss:     0.2753 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 140:  Loss:     0.2452 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 141:  Loss:     0.3063 Training Accuracy: 0.906250
    Epoch  5, Sentence Batch 142:  Loss:     0.2191 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 143:  Loss:     0.2734 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 144:  Loss:     0.2811 Training Accuracy: 0.906250
    Epoch  5, Sentence Batch 145:  Loss:     0.2783 Training Accuracy: 0.953125
    Epoch  5, Sentence Batch 146:  Loss:     0.2433 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 147:  Loss:     0.3124 Training Accuracy: 0.937500
    Epoch  5, Sentence Batch 148:  Loss:     0.2270 Training Accuracy: 0.968750
    Epoch  5, Sentence Batch 149:  Loss:     0.2422 Training Accuracy: 0.933333
    Epoch  6, Sentence Batch 0:  Loss:     0.2381 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 1:  Loss:     0.2458 Training Accuracy: 0.937500
    Epoch  6, Sentence Batch 2:  Loss:     0.2435 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 3:  Loss:     0.1785 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 4:  Loss:     0.1901 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 5:  Loss:     0.2130 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 6:  Loss:     0.2158 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 7:  Loss:     0.1752 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 8:  Loss:     0.2073 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 9:  Loss:     0.2671 Training Accuracy: 0.937500
    Epoch  6, Sentence Batch 10:  Loss:     0.2088 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 11:  Loss:     0.1677 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 12:  Loss:     0.2276 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 13:  Loss:     0.1967 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 14:  Loss:     0.1849 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 15:  Loss:     0.2009 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 16:  Loss:     0.2221 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 17:  Loss:     0.1804 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 18:  Loss:     0.2441 Training Accuracy: 0.937500
    Epoch  6, Sentence Batch 19:  Loss:     0.1925 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 20:  Loss:     0.2075 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 21:  Loss:     0.1848 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 22:  Loss:     0.2632 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 23:  Loss:     0.2219 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 24:  Loss:     0.1790 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 25:  Loss:     0.1983 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 26:  Loss:     0.1870 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 27:  Loss:     0.2229 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 28:  Loss:     0.2017 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 29:  Loss:     0.1845 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 30:  Loss:     0.2193 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 31:  Loss:     0.1948 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 32:  Loss:     0.2168 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 33:  Loss:     0.2395 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 34:  Loss:     0.2680 Training Accuracy: 0.937500
    Epoch  6, Sentence Batch 35:  Loss:     0.1790 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 36:  Loss:     0.1663 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 37:  Loss:     0.1911 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 38:  Loss:     0.2189 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 39:  Loss:     0.1963 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 40:  Loss:     0.2138 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 41:  Loss:     0.2223 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 42:  Loss:     0.2292 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 43:  Loss:     0.2146 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 44:  Loss:     0.2121 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 45:  Loss:     0.1812 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 46:  Loss:     0.1893 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 47:  Loss:     0.1802 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 48:  Loss:     0.1822 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 49:  Loss:     0.1940 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 50:  Validation Accuracy: 0.790807
    Loss:     0.2347 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 51:  Loss:     0.2040 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 52:  Loss:     0.2284 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 53:  Loss:     0.1782 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 54:  Loss:     0.1843 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 55:  Loss:     0.1641 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 56:  Loss:     0.1820 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 57:  Loss:     0.2268 Training Accuracy: 0.937500
    Epoch  6, Sentence Batch 58:  Loss:     0.2777 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 59:  Loss:     0.2273 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 60:  Loss:     0.2045 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 61:  Loss:     0.2031 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 62:  Loss:     0.1863 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 63:  Loss:     0.2015 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 64:  Loss:     0.1894 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 65:  Loss:     0.1805 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 66:  Loss:     0.2119 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 67:  Loss:     0.2064 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 68:  Loss:     0.1903 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 69:  Loss:     0.1957 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 70:  Loss:     0.1675 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 71:  Loss:     0.1831 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 72:  Loss:     0.1765 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 73:  Loss:     0.1701 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 74:  Loss:     0.1950 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 75:  Loss:     0.2211 Training Accuracy: 0.937500
    Epoch  6, Sentence Batch 76:  Loss:     0.1984 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 77:  Loss:     0.1800 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 78:  Loss:     0.2373 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 79:  Loss:     0.2396 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 80:  Loss:     0.1956 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 81:  Loss:     0.2243 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 82:  Loss:     0.1837 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 83:  Loss:     0.1731 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 84:  Loss:     0.1738 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 85:  Loss:     0.2225 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 86:  Loss:     0.2078 Training Accuracy: 0.937500
    Epoch  6, Sentence Batch 87:  Loss:     0.2436 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 88:  Loss:     0.1824 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 89:  Loss:     0.2418 Training Accuracy: 0.937500
    Epoch  6, Sentence Batch 90:  Loss:     0.2456 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 91:  Loss:     0.2549 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 92:  Loss:     0.2130 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 93:  Loss:     0.2212 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 94:  Loss:     0.1830 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 95:  Loss:     0.2331 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 96:  Loss:     0.2361 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 97:  Loss:     0.1746 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 98:  Loss:     0.2119 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 99:  Loss:     0.1821 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 100:  Loss:     0.2136 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 101:  Loss:     0.1779 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 102:  Loss:     0.1793 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 103:  Loss:     0.1814 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 104:  Loss:     0.2327 Training Accuracy: 0.937500
    Epoch  6, Sentence Batch 105:  Loss:     0.2261 Training Accuracy: 0.937500
    Epoch  6, Sentence Batch 106:  Loss:     0.2505 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 107:  Loss:     0.2770 Training Accuracy: 0.937500
    Epoch  6, Sentence Batch 108:  Loss:     0.2629 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 109:  Loss:     0.2032 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 110:  Loss:     0.1972 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 111:  Loss:     0.2653 Training Accuracy: 0.937500
    Epoch  6, Sentence Batch 112:  Loss:     0.2061 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 113:  Loss:     0.2023 Training Accuracy: 0.937500
    Epoch  6, Sentence Batch 114:  Loss:     0.2233 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 115:  Loss:     0.2177 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 116:  Loss:     0.2096 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 117:  Loss:     0.2071 Training Accuracy: 0.937500
    Epoch  6, Sentence Batch 118:  Loss:     0.1572 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 119:  Loss:     0.1800 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 120:  Loss:     0.1880 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 121:  Loss:     0.1762 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 122:  Loss:     0.1755 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 123:  Loss:     0.1870 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 124:  Loss:     0.1966 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 125:  Loss:     0.2261 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 126:  Loss:     0.2197 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 127:  Loss:     0.2243 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 128:  Loss:     0.1997 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 129:  Loss:     0.1854 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 130:  Loss:     0.2208 Training Accuracy: 0.921875
    Epoch  6, Sentence Batch 131:  Loss:     0.1710 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 132:  Loss:     0.2681 Training Accuracy: 0.921875
    Epoch  6, Sentence Batch 133:  Loss:     0.2720 Training Accuracy: 0.937500
    Epoch  6, Sentence Batch 134:  Loss:     0.1478 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 135:  Loss:     0.2123 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 136:  Loss:     0.1577 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 137:  Loss:     0.1987 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 138:  Loss:     0.1737 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 139:  Loss:     0.1574 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 140:  Loss:     0.2409 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 141:  Loss:     0.2148 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 142:  Loss:     0.1889 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 143:  Loss:     0.2154 Training Accuracy: 0.984375
    Epoch  6, Sentence Batch 144:  Loss:     0.2717 Training Accuracy: 0.906250
    Epoch  6, Sentence Batch 145:  Loss:     0.2043 Training Accuracy: 0.953125
    Epoch  6, Sentence Batch 146:  Loss:     0.2228 Training Accuracy: 0.968750
    Epoch  6, Sentence Batch 147:  Loss:     0.1805 Training Accuracy: 1.000000
    Epoch  6, Sentence Batch 148:  Loss:     0.2439 Training Accuracy: 0.921875
    Epoch  6, Sentence Batch 149:  Loss:     0.2340 Training Accuracy: 0.933333
    Epoch  7, Sentence Batch 0:  Validation Accuracy: 0.786116
    Loss:     0.1845 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 1:  Loss:     0.1547 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 2:  Loss:     0.1750 Training Accuracy: 0.937500
    Epoch  7, Sentence Batch 3:  Loss:     0.1405 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 4:  Loss:     0.1500 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 5:  Loss:     0.1896 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 6:  Loss:     0.1624 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 7:  Loss:     0.1750 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 8:  Loss:     0.1785 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 9:  Loss:     0.1680 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 10:  Loss:     0.2138 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 11:  Loss:     0.2201 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 12:  Loss:     0.1573 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 13:  Loss:     0.1800 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 14:  Loss:     0.1464 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 15:  Loss:     0.1334 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 16:  Loss:     0.1530 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 17:  Loss:     0.1806 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 18:  Loss:     0.1527 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 19:  Loss:     0.1919 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 20:  Loss:     0.1578 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 21:  Loss:     0.1655 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 22:  Loss:     0.1462 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 23:  Loss:     0.1508 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 24:  Loss:     0.1356 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 25:  Loss:     0.1801 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 26:  Loss:     0.2241 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 27:  Loss:     0.1440 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 28:  Loss:     0.1945 Training Accuracy: 0.953125
    Epoch  7, Sentence Batch 29:  Loss:     0.1559 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 30:  Loss:     0.1460 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 31:  Loss:     0.1500 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 32:  Loss:     0.1596 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 33:  Loss:     0.1639 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 34:  Loss:     0.1522 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 35:  Loss:     0.1500 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 36:  Loss:     0.1798 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 37:  Loss:     0.1969 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 38:  Loss:     0.2239 Training Accuracy: 0.953125
    Epoch  7, Sentence Batch 39:  Loss:     0.1879 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 40:  Loss:     0.1548 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 41:  Loss:     0.1257 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 42:  Loss:     0.1359 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 43:  Loss:     0.2179 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 44:  Loss:     0.1342 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 45:  Loss:     0.1594 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 46:  Loss:     0.2085 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 47:  Loss:     0.1699 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 48:  Loss:     0.1777 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 49:  Loss:     0.1426 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 50:  Loss:     0.1377 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 51:  Loss:     0.1484 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 52:  Loss:     0.1797 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 53:  Loss:     0.2017 Training Accuracy: 0.937500
    Epoch  7, Sentence Batch 54:  Loss:     0.1566 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 55:  Loss:     0.1451 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 56:  Loss:     0.1492 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 57:  Loss:     0.1706 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 58:  Loss:     0.1207 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 59:  Loss:     0.1780 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 60:  Loss:     0.1961 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 61:  Loss:     0.1469 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 62:  Loss:     0.1375 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 63:  Loss:     0.2146 Training Accuracy: 0.937500
    Epoch  7, Sentence Batch 64:  Loss:     0.1663 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 65:  Loss:     0.1623 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 66:  Loss:     0.1237 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 67:  Loss:     0.1958 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 68:  Loss:     0.1911 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 69:  Loss:     0.1770 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 70:  Loss:     0.1601 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 71:  Loss:     0.1581 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 72:  Loss:     0.1459 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 73:  Loss:     0.1915 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 74:  Loss:     0.1492 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 75:  Loss:     0.1848 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 76:  Loss:     0.1500 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 77:  Loss:     0.1754 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 78:  Loss:     0.1585 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 79:  Loss:     0.1451 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 80:  Loss:     0.2094 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 81:  Loss:     0.1482 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 82:  Loss:     0.1746 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 83:  Loss:     0.2164 Training Accuracy: 0.953125
    Epoch  7, Sentence Batch 84:  Loss:     0.1690 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 85:  Loss:     0.1773 Training Accuracy: 0.953125
    Epoch  7, Sentence Batch 86:  Loss:     0.1418 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 87:  Loss:     0.1580 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 88:  Loss:     0.1960 Training Accuracy: 0.953125
    Epoch  7, Sentence Batch 89:  Loss:     0.1614 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 90:  Loss:     0.1829 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 91:  Loss:     0.1564 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 92:  Loss:     0.1295 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 93:  Loss:     0.1599 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 94:  Loss:     0.1601 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 95:  Loss:     0.1527 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 96:  Loss:     0.1567 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 97:  Loss:     0.1634 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 98:  Loss:     0.1646 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 99:  Loss:     0.1437 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 100:  Validation Accuracy: 0.786116
    Loss:     0.1721 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 101:  Loss:     0.1898 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 102:  Loss:     0.1602 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 103:  Loss:     0.1756 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 104:  Loss:     0.1579 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 105:  Loss:     0.1576 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 106:  Loss:     0.1845 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 107:  Loss:     0.1539 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 108:  Loss:     0.2031 Training Accuracy: 0.937500
    Epoch  7, Sentence Batch 109:  Loss:     0.1742 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 110:  Loss:     0.1580 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 111:  Loss:     0.1512 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 112:  Loss:     0.1439 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 113:  Loss:     0.1599 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 114:  Loss:     0.1638 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 115:  Loss:     0.1382 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 116:  Loss:     0.2040 Training Accuracy: 0.953125
    Epoch  7, Sentence Batch 117:  Loss:     0.1508 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 118:  Loss:     0.2179 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 119:  Loss:     0.1397 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 120:  Loss:     0.1431 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 121:  Loss:     0.1592 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 122:  Loss:     0.2031 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 123:  Loss:     0.1785 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 124:  Loss:     0.1502 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 125:  Loss:     0.1961 Training Accuracy: 0.937500
    Epoch  7, Sentence Batch 126:  Loss:     0.1257 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 127:  Loss:     0.1692 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 128:  Loss:     0.1766 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 129:  Loss:     0.1782 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 130:  Loss:     0.1782 Training Accuracy: 0.953125
    Epoch  7, Sentence Batch 131:  Loss:     0.1610 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 132:  Loss:     0.1360 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 133:  Loss:     0.1566 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 134:  Loss:     0.2191 Training Accuracy: 0.953125
    Epoch  7, Sentence Batch 135:  Loss:     0.2002 Training Accuracy: 0.937500
    Epoch  7, Sentence Batch 136:  Loss:     0.1736 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 137:  Loss:     0.1750 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 138:  Loss:     0.1726 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 139:  Loss:     0.1369 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 140:  Loss:     0.1447 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 141:  Loss:     0.1489 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 142:  Loss:     0.1778 Training Accuracy: 0.953125
    Epoch  7, Sentence Batch 143:  Loss:     0.1467 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 144:  Loss:     0.1733 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 145:  Loss:     0.1284 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 146:  Loss:     0.1414 Training Accuracy: 1.000000
    Epoch  7, Sentence Batch 147:  Loss:     0.2057 Training Accuracy: 0.984375
    Epoch  7, Sentence Batch 148:  Loss:     0.1793 Training Accuracy: 0.968750
    Epoch  7, Sentence Batch 149:  Loss:     0.1506 Training Accuracy: 0.966667
    #######VALIDATION STATS#######
    Validation Accuracy: 0.791745



```python

```
