## Lab assignment 1

### Luka van der Plas (4119142) & Shu Zhao (6833519)

#### Question 1: Can you think of another application where automatic recognition of hand-written numbers would be useful?

1. The OCR (Optical Character Recognition) system which can recognize the numeric entries in the application forms filled up by hand in any bank branch, e.g. the applicant's bank account. And with the help of OCR, the paper forms can be automatically scanned and logged in the database.

2. It would be useful for digitalising instance historic records (though these would often require their own training data to account for historic handwriting). Optical character recognition can allow searching through text, etc.

#### Question 2: In the output text in your console, how long did each epoch take to run?

The duration for each epoch is as below:
* Epoch 1: 5s
* Epoch 2: 3s
* Epoch 3: 4s
* Epoch 4: 3s
* Epoch 5: 3s
* Epoch 6: 3s
* Epoch 7: 4s
* Epoch 8: 4s
* Epoch 9: 3s
* Epoch 10: 3s
* Epoch 11: 4s
* Epoch 12: 3s

The console screen shot is as below:
![epoch-time-elapse](./pix/epoch-time-elapse.png)

#### Question 3: Plot the training history and add it to your answers.

The training history is as below:
![plot-of-history-1](./pix/plot-of-history-1.png)

#### Question 4: Describe how the accuracy on the training and validation sets progress differently across epochs, and what this tells us about the generalisation of the model. 

Around the epoch 4, the accuracy rate for the validation set reaches its maximum value 0.927 and begins to degrade and stagnate at approximately 0.925, which means that further changes to the model are no longer increasing its fit to new data. The accuracy rate for the training set continues to increase until it stalls around 0.925, which means the model can no longer find a better fit, and the remaining errors are the result of the limitations of the model and the dataset.

The scores for the training and validation set over the last few epochs are nearly identical, which suggests that the model generalises well to new data.

#### Question 5: What values do you get for the model’s accuracy and loss? 

The model's accuracy is 0.9217.

The model's loss is 0.290271.

#### Question 6: Discuss whether this accuracy is sufficient for some uses of automatic hand-written digit classification. 

This accuracy is sufficient, based on the following reasons:
1. For the hand-written digit classification, the random accuracy should be 0.1, whereas this model generates the accuracy 0.9217. So this model achieves the statistical power.

This accuracy is insufficient, based on the following reasons:
1. It is not commercially sufficient, as the error rate is 7.83%, and for important documents like bank cheques, we don't want any errors. However, a 90% accuracy for individual digits may not be as bad as it sounds. If a system has to register a bank account number, for instance, an error in a single digit will result in an invalid number - something which a computer can also easily verify. As such, the chances of a computer registering an incorrect bank account number without raising an error would be much lower, since it needs to make multiple errors and they need to cancel each other out.

#### Question 7:  How does linear activation of units limit the possible computations this model can perform?

When an input value x contributues to the activation A of a unit, a linear activation does not allow to implement some threshold for x. This means that very low values of x still contribute to A, even if those are not actually important. This leads to more noise in the data, which makes training harder.

#### Question 8: Plot the training history and add it to your answers.

The training history is as below:
![plot-of-history-2](./pix/plot-of-history-2.png)

#### Question 9: How does the training history differ from the previous model, for the training and validation sets? What does this tell us about the generalisation of the model?

The accuracy rate of this training history achieves 0.979 for the validation set and 0.9966 for the training set respectively, which are much higher than 0.927 and 0.925 from the previous model. Moreover, the accurary rate for the validation set keeps increasing until it peaks at epoch 9 with the value 0.979, whereas in the previous model for the validation set, it has a pre-mature stop at epoch 4.

The high accuracy for the validation set tells that the generalization of this model is pretty good, and it will perform better on new data than the previous model. However, we also see that the accuracy on the valuation set stagnates around epoch 6, while the accuracy on the training set continues to increase. This means that at this point, the model is finding a better fit for the training set without increasing its generalisation value, which means that it is overfitting.

#### Question 10: How does the new model’s accuracy on test set classification differ from the previous model? Why do you think this is?

The new model's accuracy on the test set is 0.9814, which is much higher than the accuracy from the previous model. Apparently the ReLU activation improves the model's generalisation, because it is less sensitive to noise in the new data, due to the activation threshold.

#### Question 11: Plot the training history and add it to your answers.

The training history is as below:
![plot-of-history-3](./pix/plot-of-history-3.png)

#### Question 12: How does the training history differ from the previous model, for the training and validation sets? What does this tell us about the generalisation of the model?

For both the training set and the validation set, the training history remains almost the same as that of the previous model. What is slightly different is for the validation set. The accurary for the validation set starts at 0.9809 from epoch 1, and it fluctuates slightly since then around 0.99 with even a minor drop at the last epoch to 0.9858, though this may be random fluctuation.

The generalisation of the model is adequate since it performs very well on the training set, but there seems to be some overfitting going on in the last few epochs, as the model improves itself continuously on the training set, but stays numb to the validation set. 

#### Question 13: What values do you get for the model’s accuracy and loss? 

The model's accuracy is 0.9896.

The model's loss is 0.03824518.

#### Question 14: Discuss whether this accuracy is sufficient for some uses of automatic hand-written digit classification.

This accuracy is sufficient, based on the following reasons:
1. For the hand-written digit classification, the random accuracy should be 0.1, whereas this model generates the accuracy 0.9896, which is much higher than the score of the previous model. So this model achieves stronger statistical power.

2. It is commercially sufficient also, as the error rate of this model is 1.04%.

#### Question 15: Describe the principles of overfitting and how dropout can reduce this.

Overfitting happens when the model creates a very close fit to the training data, which matches all the specific fluctuations of the training data, even ones that are not relevant. This decreases the generalisation of the model.
Dropout prevents the model from aligning too closely with the training set, by introducing arbitrary noises into the training set during training.

#### Question 16: How does the training history differ from the previous (convolutional) model, for both the training and validation sets, and for the time taken to run each model epoch?

For both the training set and the validation set, the accurary keeps upgrading. For the training set, the accurary witnesses a steep line from 0.9039 to 0.9862. And for the validation set, it increases slightly from 0.9781 to 0.9889. What differs from the previous model is that the results for the validation set are very close to those of the training set.

The training history is as below for this new model:
![plot-of-history-4](./pix/plot-of-history-4.png)

For the time taken to train the model, the previous model takes 80 seconds per epoch on average, compared with this model's 103 seconds per epoch on average.

#### Question 17: What does this tell us about the generalisation of the two models? 

It tells that the previous model without dropout is less general than the current model with dropout. It can be deduced that the dropout layer can increase generalization and prevent overfitting during the training of models.

#### Question 18: What code did you use to define the model described here?

```
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu', input_shape = c(32, 32, 3), padding = "same") %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu', padding = "same") %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = 'softmax')
```

#### Question 19: Execute this model fit command. After your fitting is finished, plot the training history and put it in your answers.

The training history is as below:
![cifar10](./pix/cifar10.png)

#### Question 20: How does the training history differ from the convolutional model for digit recognition? Why do you think this is?

The difference of the training history focuses on below two points:

1. **Validation accuracy**: This model's validation accuracy is 0.7196, which is much lower than the previous model's validation accuracy 0.9889.

2. **Growth of validation and training accuracy**: For this model, the validation and training accuracy both start at a low value, which is 0.4267 and 0.3137 respectively, and the accuracy increases gradually as the training progresses. But for the previous model, the validation and training accuracy both start at a high value, which is above 0.9. More specifically, the validation accuracy is 0.975 at epoch 1, while the training accuracy increases rapidly from epoch 1 to epoch 2, from around 0.9 to above 0.96. Afterwards they just grow to a very limited extent.

As to the reason why this difference occurs:

1. **Dataset**: This model's dataset cifar10 is of the shape (32, 32, 3), which includes colored images, whereas the previous model's dataset mnist is of the shape (28, 28, 1), which just consists of black-and-white figures. As the shape shows, the cifar data is of a much higher dimension (32 * 32 * 3 dimensions, whereas the mnist data has 28 * 28 dimensions). This makes the learning more complex and increases the computational load.

2. **Multi-class classification goal**: This model's goal is to classify the colored images into ten classes, which are airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck. Because these are pictures of objects, there is variation in orientation, light, et cetera. This makes these classes harder to distinguish than the numbers in the mnist data, since they do not easily follow from the retinal image.

#### Question 21: How does the time taken for each training epoch differ from the convolutional model for digit recognition? Give several factors that may contribute to this difference.

For this model, the time taken for each training epoch is 160 seconds on average, whereas for the previous model, it is 103 seconds on average. So this model takes longer time for each training epoch to complete than the previous model.

The contributing factors are as below:

1. **Dataset**: There are two more color channels in cifar10, so it requires more computational time for each training epoch.

2. **Layers**: There are two more convolutional layers, which means more weights to train, so it takes more time for each batch to complete the training.

3. **Batch Size**: The batch size is 32 for this model, whereas it is 128 for the previous model. So for this model, it takes more batches for one epoch, which means more calculations, because the loss gradient is calculated per batch.

#### Question 22: Read the research paper “Performance-optimized hierarchical models predict neural responses in higher visual cortex”, available from: [http://www.pnas.org/content/pnas/111/23/8619.full.pdf](http://www.pnas.org/content/pnas/111/23/8619.full.pdf). Write a short (~500 word) summary of the experimental approach and results.

Humans can recognise objects despite drastic variation in the retinal image (due to lighting, position, angle, etc.). The processing of visual information that underlies this ability is performed in the ventral visual stream. This collection of brain areas can be seen as a series of processing stages, that represent visual information in increasingly more abstract ways. The lowest area, the V1, is most directly based on the retinal image and can be modelled with high accuracy. Modelling higher areas has proven more difficult. The aim of this study is to model the IT, the highest ventral cortical area.

The study constructed a large test of objects shown with varying orientations, backgrounds, etc., and gathered neural responses to these images from human participants. To investigate how model performance relates to neural predictivity, the study then tried a large number of convolutional neural networks, and evaluated their ability to recognise objects in the test set, as well as their predictivity of the measured neural activty in IT. These two measures were found to have a positive correlation. When model parameters were optimised for object recognition, this increased their IT predictivity, even when neural data was not used in optimisation.

Next, the study used activity in IT and V4 (the highest area before IT) to predict object identity in the dataset and compared it to human performance, as well as the performance of various models. IT activity roughly matches human performance in accuracy, performing well even on test sets with high variation in object orientation, lighting, etc. The predictivity of V4 greatly deteriorates with more variation in the test set. The most robust artificial networks that are tested perform similarly to V4.

The study then optimised and trained a final model for object recognition. This model is more complex than the standard three-layer CNN, using a combination of deeper CNNs with more specialised functions, using HMO to develop the architecture of the network. This model performed similar to the IT neural data in the visual recognition task, even with high variation.

The model was further evaluated by measuring its IT predictivity in different layers. Each subsequent layer of the model achieved better predictivity of neural activity in IT, and were increasingly more robust under variation in object pose and position. The highest layer in the model showed very high predictivity of IT compared to existing models.

The model’s predictivity of neural data was further evaluated. Previous comparisons had been to the activity in individual neurons, but the model was also used to predict representations of activity in the entire population. Again, predictivity of the model was high. The model’s second highest layer was also used to predict V4 activity, and was found to perform well at this task. This suggests that the internal structure of the model somewhat mirrors the internal structure of the ventral stream.

The experiment demonstrates that performance-optimised hierarchical models predict neural responses in higher visual cortex, and it also indicates that the top-down perspective can complement the bottom-up approach to understand how the visual neurons work, as lower visual cortex might be selected precisely by higher visual cortex to support its computation.

#### Question 23: Play around with these settings and see how they affect your ability to learn classification of different data sets. Write down what you found and how you interpret the effects of these settings.

The key factors that affect the accuracy of classification for different data sets are as below:
1. Input features: these are really important. More input features increases the  size of the hypothesis space. Also, if you don't use all input features, the choice of features can really affect how accurate the program can be. This really depends on the dataset. Well-chosen input features can increase the accuracy of the program and reduce the number of hidden layers necessary. 

2. Number of hidden layers & number of neurons: both of these can increase accuracy, but it's hard to see the interaction between the two: is it better to have more neurons or to add more layers? We did observe that it's not useful for a hidden layer to have more neurons than the previous one. For instance, here we try to do the same training exercies with a layer of 6 neurons followed by one with 3 neurons, or with the order of hidden layers reversed.

3-6             |  6-3
:-------------------------:|:-------------------------:
![3-6](./pix/3-6.png)  |  ![6-3](./pix/6-3.png)

The model performs significantly better when there are more neurons in the first layer instead of the second. This makes sense, because in the second case, the output of layer 1 is limited to 3 dimensions. Therefore, the increased dimensionality of layer 1 can't help in a more complex image of the dataset, because it is working with limited input.

3. Activation function: Different activation functions affect the accuracy of the result. Especially in very simple networks, it makes a big difference. Linear activation tends to perform the worst.

4. Noise: Adding noise decreases the final accuracy, but it does not really affect the training development.

Learning rate, batch size and ratio of training to test data don't necessarily affect the final result. They do have an effect the development in training, and the duration of training.

#### Question 24: What is the minimum you need in the network to classify the spiral shape with a test set loss of below 0.1?

The simplest network we could build used all input features, 1 hidden layer with 6 neurons, and ReLU activation.

![spiral](./pix/spiral.png)

#### Question 25: Write a simple function that achieves the convolution operation efficiently for two- dimensional and three-dimensional inputs. This should allow you input a set of convolutional filters (‘kernels’ in Keras’s terminology) and an input layer (or image) as inputs. The input layer should have a third dimension, representing a stack of feature maps, and each filter should have a third dimension of corresponding size. The function should output a number of two-dimensional feature maps corresponding to the number of input filters, though these can be stacked into a third dimensional like the input layer. Give your code as the answer.

The source code is in this [link](./source-code/convolution_layer.R).

#### Question 26: Write a simple function that achieves rectified linear (relu) activation, with a threshold at zero. Give your code as the answer.

The source code is in this [link](./source-code/relu.R).

#### Question 27: Write a simple function that achieves max pooling. This should allow you to specify the spatial extent of the pooling, with the size of the output feature map changing accordingly. Give your code as the answer.

The source code is in this [link](./source-code/max_pooling.R).

#### Question 28: Write a simple function that achieves normalisation within each feature map, modifying the feature map so that its mean value is zero and its standard deviation is one. Give your code as the answer. 

The source code is in this [link](./source-code/normalisation.R).

#### Question 29: Write a function that produces a fully-connected layer. This should allow you to specify the number of output nodes, and link each of these to every node a stack of feature maps. The stack of feature maps will typically be flattened into a 1- dimensional matrix first.

The "layer_flatten" source code is in this [link](./source-code/flatten.R).

The "layer_dense" source code is in this [link](./source-code/dense_layer.R).

#### Question 30: Write a function that converts the activation of a 1-dimensional matrix (such as the output of a fully-connected layer) into a set of probabilities that each matrix element is the most likely classification. This should include the algorithmic expression of a softmax (normalised exponential) function.

The source code is in this [link](./source-code/softmax.R).

#### Question 31: Explain the principle of backpropagation of error in plain English. This can be answered with minimal mathematical content, and should be IN YOUR OWN WORDS. What is backpropagation trying to achieve, and how does it do so?

Backpropagation adjusts the neural network by comparing the output of the network with the desired output. To determine where to make adjustments, the model calculates the error in the output and estimates how it changes based on the intput: if a larger input leads to a smaller error, we want the network to give a higher weight to that input. The error is estimated by comparing the actual output of the network with its desired output (i.e. the output in the training data).
When making changes, the changes to a node in the network are proportional to the steepness of the error curve if we map the input of that node to the error of the network. The size of these adjustments is also dependent upon the learning rate of the model: a higher learning rate leads to more rigorous changes. 

#### Question 32 (BONUS QUESTION): Describe the process of backpropagation in mathematical terms. Here, explain (in English) what each equation you give does, and relate this to the answers given in Question 31. You are welcome to express equations in R code (not python) rather than using equation layout.

We have a network with input X ( = [X1, X2 .... ]) and output y'. We compare that to the desired output y.

The error of the network is estimated by a loss function, which may vary depending on the problem. For instance, we can use say E = (y - y')^2 .
 
 Estimating how the error changes as a result of weight in the network w is done by calculating the derivative (d E / d w ). This is derived by appying the chain rule:
 (d E / d w ) = (d E / y' ) ) ( d y' / d w )

We now want to calculate two values: the derivative of the error as a function of the output, and the derivative of the output as a function of the weight. The exact formulas for these depend on the type of network. When we have the derivative of the error as a function of the weight (so we know how to adjust the weight to minimise the error), we adjust the weight as follows:

D w = - r (d E / d w)

where r is the learning rate of the training model. The new value of w, which we will call w', is equal to w + D w.

#### Question 33 (BONUS QUESTION): Write a function to achieve backpropagation of error to affect the convolutional filter (kernel) structure used in question 25. Modify your function from question 25 to give the filters used as an output, so you can modify these filters using backpropagation. Initialise the network with random weights in the filters. Give the code for your convolution and backpropagation functions as your answer.

We were a bit confused by the question, because a backpropagation algorithm requires more than a single convolutional filter (you need the entire network). We wrote a backpropagation algorithm for our neural network with 2 hidden layers that we implemented in question 35.

The code below defines a backpropagation function. This function takes the parameters of the network as input, together with a single input image, its target output and the learning rate. As output, it gives the adapted parameters based on the loss and learning rate. We implement this backpropagation function in question 36, so that is where you can see how it can be used.

The source code is in this [link](./source-code/back_propagation.R).

#### Question 35 (BONUS QUESTION): Write a piece of code that uses all of these functions (Questions 25-33) together to make a convolutional neural network with two convolutional layers, a fully connected layer, and an output layer (pooling is optional, but thresholding and normalisation are required). This should give the accuracy of the labels as an output. Give your code as your answer. 

The source code is in this [link](./source-code/forward_propagation.R).

#### Question 36 (BONUS QUESTION): Use the resulting function to learn to classify the mnist data set, as you did in question 11. Plot the progression of the classification accuracy over 100 cycles.

We created the model, and wrote a loop that adapts the model using stochastic gradient descent. However, it takes a long time to evaluate a single mnist image, so we only evaluated 10 images.
The loss over the 10 images is plotted below:
![pix](./pix/loss.png)

The source code is in this [link](./source-code/classification.R).
