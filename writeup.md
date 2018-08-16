# **Traffic Sign Recognition** 

## Writeup

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

First I investigated how many instances of different sign types are in training set. Model will probably
bias classifying uncertain instances as signs that have more instances in training set. Also sign types with
less instances will probably have worse performance.

This was the result:

![training instances](./img/trainin_instances_by_sign.png)

This clearly shows that there are a lot of variability and some sign types don't have enough instances and it would
be good to generate additional instances.

To have better understanding of different sign types, I plotted random instance from each sign type:

![sign types](./img/sign_types.png)

We can notice following things from these samples:
* Image resolution is low and some signs are hard to classify even for human
* Some sign types are very similar, so model will probably struggle to predict these:
    * red triangle with black specifics in middle, for ex. 23, 24, 27, 28
    * red circle with specifics in middle, for ex. 0, 1, 2, 4, 9, ...
    * some images have bad lighting conditions

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I used iterative approach by changing one aspect of neural net architecture and then testing change with re-training network and measuring performance with validation set.

##### 1. First version, LeNet
First architecture was just LeNet from previous lab. Image data was not preprocessed, no grayscaling or normalisation was done. This gave baseline performance that was used that future changes to preprocessing and model actually improved performance.

10 epochs were used to train the model.

Validation set accuracy: **87.3%**

##### 2. Normalization

First change to pipeline was adding normalization. All training, validation and testing image data was normalized by 2 ways:
* **Centering**: data was centered around mean 0, which improves vanishing/exploding gradients and also increased convergence
* **Scaling**: Image data was scaled down by dividing with 1 standard deviation, which helps convergence speed and accuracy

20 epochs were used. At first I tried with 10 epoch, but because validation accuracy kept improving, I increased epochs to 20.

Validation set accuracy: **92.7%**

##### 3. Grayscale

Improved data pipeline adding image conversion to grayscale. This didn't change accuracy.

Validation set accuracy: **92.4%**

##### 4. Dropout

To avoid overfitting, dropout with keep probability 0.5 was added to dense (fully connected) layers. Dropout was not added to convolutional layers as weights are shared between spatial positions so there shouldn't be huge number of parameters to overfit.

Validation set accuracy: **95.1%**

##### 5. Additional layers

Added new convolutional and dense layers, but this didn't improve performance.

Validation set accuracy: **95.2%**

##### 6. L2 regularization

Added L2 regularization to all layers, this should reduce overfitting. 

Tried with different beta values, but settled to 0.001.

Validation set accuracy: **96.2%**

##### 7. Data augmentation

Added data augmentation to data pipeline by generating 3000 new images to every sign type by rotating, translating, shearing existing samples.

As these images were generated in data pipeline and not while training, I was limited to memory available. As this didn't improve performance , I didn't look into improving this.
 
Validations set accuracy: **95.6%**

![sign types](./img/data%20augmentation.png)


##### 8. More filters

As model seems to have stagnated, I suspected that I need to have more parameters, so increased filter count in convolutional layers to 64. Tried different values like 32, but 64 gave best performance.

Validation set accuracy: **98.1%**

##### 9. Test set accuaracy

As I had good enough validation set accuracy, it was time to test model with test data set.

Test set accuracy: **95.2%**

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

