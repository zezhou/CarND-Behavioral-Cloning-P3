# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points]

(https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of two convolution neural network with 3x3 filter sizes and depths 12 (model.py lines 93,95). The model also includes two Max Pooling with 2x2 filter and 2x2 strides(model.py lines 94,96).

The model includes RELU layers to introduce nonlinearity (code line 90, 92, 95), and the data is normalized in the model using a Keras lambda layer (code line 88). Besides, The training images is cropping before training (code line 92). The model also includes Batch Normalization (code line 99) to accelerate training. 

#### 2. Attempts to reduce overfitting in the model
The model contains dropout layers in order to reduce overfitting (model.py lines 100). The dropout rate is set to 50%. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 80). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 99).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Besides, I record extra data in difficult roads, especially roads the model failed in. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to balance model capacity and training speed in my laptop. 

My first step was to use a convolution neural network model similar to the vgg. I thought this model might be appropriate because it has good capacity and fast to train.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model with introducing dropout layer so that the overfitting is hugely decreasing.

Then I introduce batch normalization to continue decrease overfitting and improve training speed.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Most of them were corner. To improve the driving behavior in these cases, I drive car pass these spots in training mode, and record these logs to generate training datasets.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 89-104) consisted of a convolution neural network with the following layers and layer sizes .

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:| 
| Input                 | 160x320x3 RGB image                           |
| Lambda                | Normalize input                               |
| Cropping2D            | cropping input image                          |
| Convolution 3x3       | 3x3 stride, same padding                      |   
| RELU                  | ReLU - a rectified linear unit                |
| Max pooling           | 2x2 stride                                    |
| Convolution 3x3       | 3x3 stride, same padding                      |
| RELU                  | ReLU - a rectified linear unit                |
| Max pooling           | 2x2 stride                                    |
| Flatten               |                                               |
| Fully connected       | 50                                            |
| RELU                  | ReLU - a rectified linear unit                |
| BatchNormalization    |                                               |
| Dropout               | keep probability = 0.5                        |
| Fully connected       | 1 Dense                                       |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

[center_image]: ./examples/center_2017_10_17_18_40_24_650.jpg "center image"
![center imgage][center_image]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to back to the track. These images show what a recovery looks like starting from left to center and right to center :

[left_to_center_image]: ./examples/left_2017_10_17_18_40_24_650.jpg "left to center image"
[right_to_center_image]: ./examples/right_2017_10_17_18_40_24_650.jpg "right to center image"

![left to center image][left_to_center_image]
![right to center image][right_to_center_image]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would increase the number of training samples to make model more robust. For example, here is an image that has then been flipped:

[flip_image]: center_2017_10_17_18_40_24_650_flip.jpg "flip image"

![flip image][flip_image]

After the collection process, I had *24108* number of data points. I then preprocessed this data by normalizing and cropping.

I finally randomly shuffled the data set and put *25%* of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the mse of validation is increasing after 6 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
