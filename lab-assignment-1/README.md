## Lab assignment 1


#### Question 1: Can you think of another application where automatic recognition of hand-written numbers would be useful?

1. The OCR (Optical Character Recognition) system which can recognize the numeric entries in the application forms filled up by hand in any bank branch, e.g. the applicant's bank account. And with the help of OCR, the paper forms can be automatically scanned and logged in the database.

2. License plate of cars?


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

Around the epoch 4, the accuracy rate for the validation set reaches its maximum value 0.927 and begins to degrade and stagnate at approximately 0.925, whereas the accuracy rate for the training set continues to increase until it stalls around 0.925.

It tells that the generalisation of the model is not good, as the model is over-fit for the training set and lacks the capability to predict the unknown patterns in the validation set.


#### Question 5: What values do you get for the model’s accuracy and loss? 

The model's accuracy is 0.9217.

The model's loss is 0.290271.


#### Question 6: Discuss whether this accuracy is sufficient for some uses of automatic hand-written digit classification. 

This accuracy is sufficient, based on the following reasons:
1. For the hand-written digit classification, the random accuracy should be 0.1, whereas this model generates the accuracy 0.9217. So this model achieves the statistical power.

This accuracy is insufficient, based on the following reasons:
1. It is not commercially sufficient, as the error rate is 7.83%, which is above 1%.


#### Question 7:  How does linear activation of units limit the possible computations this model can perform?

The linear activation of units makes the model more computationally expensive and may lead to the overfitting.


#### Question 8: Plot the training history and add it to your answers.

The training history is as below:
![plot-of-history-2](./pix/plot-of-history-2.png)


#### Question 9: How does the training history differ from the previous model, for the training and validation sets? What does this tell us about the generalisation of the model?

The accuracy rate of this training history achieves 0.979 for the validation set and 0.9966 for the training set respectively, which are much higher than 0.927 and 0.925 from the previous model. Moreover, the accurary rate for the validation set keeps increasing until it peaks at epoch 9 with the value 0.979, whereas in the previous model for the validation set, it has a pre-mature stop at epoch 4.

It tells that the generalization of this model is pretty good, though it performs worse on the validation set compared with accuracy rate from the training set, but the accuracy rate for the validation set keeps growing and it reaches higher accuracy rate with 0.979.


#### Question 10: How does the new model’s accuracy on test set classification differ from the previous model? Why do you think this is?

The new model's accuracy on the test set is 0.9814, which is much higher than the accuracy from the previous model.

It is because the generalization of this new model is good, so it can predict more accurately for the unknown patterns in the test set.


#### Question 11: Plot the training history and add it to your answers.

The training history is as below:
![plot-of-history-3](./pix/plot-of-history-3.png)


#### Question 12: How does the training history differ from the previous model, for the training and validation sets? What does this tell us about the generalisation of the model?

For both the training set and the validation set, the training history remains almost the same as that of the previous model. What is slightly different is for the validation set. The accurary for the validation set starts at 0.9809 from epoch 1, and it fluctuates slightly since then around 0.99 with even a minor drop at the last epoch to 0.9858.

It tells that the generalisation of the model is not good, as the model improves itself continuously on the training set, but stays numb to the validation set. 


#### Question 13: What values do you get for the model’s accuracy and loss? 

The model's accuracy is 0.9896.

The model's loss is 0.03824518.


#### Question 14: Discuss whether this accuracy is sufficient for some uses of automatic hand-written digit classification.

This accuracy is sufficient, based on the following reasons:
1. For the hand-written digit classification, the random accuracy should be 0.1, whereas this model generates the accuracy 0.9896, which is much higher than the score of the previous model. So this model achieves stronger statistical power.

2. It is commercially sufficient also, as the error rate of this model is 1.04%.


#### Question 15: Describe the principles of overfitting and how dropout can reduce this.

The principle of overfitting is the tradeoff between optimization and generalization for the model over the training history. As the learning progresses, the model gains optimization which aligns too well with the training set, while it loses generalization to predict the unforseen data from the validation set.

The dropout can remove a certain fraction of the features from the output during the training. Thus the model will not align too closely with the training set, as these arbitary noises are introduced into the training set.


#### Question 16: How does the training history differ from the previous (convolutional) model, for both the training and validation sets, and for the time taken to run each model epoch?

For both the training set and the validation set, the accurary keeps upgrading. For the training set, the accurary witnesses a steep line from 0.9039 to 0.9862. And for the validation set, it increases slightly from 0.9781 to 0.9889. What differs from the previous model is that the model reaches a higher accurary for the validation set than for the training set.

The training history is as below for this new model:
![plot-of-history-4](./pix/plot-of-history-4.png)

For the time taken to run the model, the previous model takes 80 seconds per epoch on average, compared with this model's 103 seconds per epoch on average.


#### Question 17: What does this tell us about the generalisation of the two models? 

It tells that the previous model without dropout is less general than the current model with dropout. It can be deduced that the dropout layer can increase generalization and prevent overfitting during the training of models.



