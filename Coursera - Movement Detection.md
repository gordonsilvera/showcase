## Coursera: Machine Learning Final Project
Gordon Silvera

7/25/15

R Code: https://github.com/gordonsilvera/coursera/blob/master/machine-learning.R

###About the Dataset
This dataset tracks various forms of movements while an individual lifts dumbbells. The outcome variable, "classe", is the type of **Dumbbell Biceps Curl** in five different fashions:
- Class A: exactly according to the specification 
- Class B: throwing the elbows to the front 
- Class C: lifting the dumbbell only halfway
- Class D: lowering the dumbbell only halfway
- Class E: throwing the hips to the front

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3gv6LfTpo

To cleanse the data, I have removed variables (features) with missing data. With the remaining features, I realized that they were aggregated into two types of groups. To simply certain portions of the analysis, I will analyze data within these "sub-feature" groupings. 
- *Body Dimensions*: belt, arm, dumbbell, forearm
- *Metric Dimensions*: roll, pitch, yaw, total acceleration, gyration (x,y,z), acceleration (x,y,z), magnet (x,y,z)


###Cross Validation
For cross validation, I have randomly selected 75% of “pml-training.csv” to train my models. I used the following code snippet to do so:
```
inTrain <- createDataPartition(y=mainTrain1$classe, p=0.75, list=FALSE)    # mainTrain1 has null values removed
training <- mainTrain1[inTrain,]
testing <- mainTrain1[-inTrain,]
dim(training); dim(testing)
```

###Exploratory Data Analysis
Initially, I would like to better understand my variables, as well as how they related to one another and my outcome variable. I first applied the `summary(mainTrain1)` function on the dataset. From there, I started considering the relationships between the explanatory variables with Q-plots and Density plots.
```
qplot(pitch_dumbbell, total_accel_dumbbell, colour=classe, data=training)   # sample Q-plot
qplot(total_accel_forearm, colour=classe, data=training, geom="density")    # sample Density Plots
```

###Model Development

#####Initial Model
At first, I created a Linear Discriminant Analysis model that included all numerical features without pre-processing. This generated an Accuracy of 69.9%. I will use this as a benchmark going forward. 

```
trainingInput <- training[,(7:59)]
modelFit0 <- train(classe~., data = trainingInput, method = "lda")
modelFit0
```

#####Principle Components Analysis
I then considered relationships between the explanatory features using Principle Componnets. I started by finding the most correlated variables among my potential features. 
```
M <- abs(cor(training[,7:58]))
diag(M) <- 0
which(M > 0.8, arr.ind = T)
```
This output revealed several correlated features. I've grouped these into "sets" below.
- *Set 1*: yaw_belt, total_accel_belt, accel_belt_y, accel_belt_z, roll_belt
- *Set 2*: accel_belt_x, magnet_belt_x, pitch_belt
- *Set 3*: gyros_arm_x, gyros_arm_y
- *Set 4*: accel_arm_x, magnet_arm_x, magnet_arm_y, magnet_arm_z
- *Set 5*: pitch_dumbbell, yaw_dumbbell, accel_dumbbell_x, accel_dumbbell_z
- *Set 6*: gyros_dumbbell_x, gyros_dumbbell_z, gyros_forearm_z, gyros_forearm_y

Next, I developed a model with principle components with the `preProcess()` function (see model below). However, this generated an accuracy of 52% versus 70% for a model without PCA. 
```
modelFit1 <- train(classe~., data = trainingInput, method = "lda", preProcess=c("pca"))
modelFit1
```
Therefore -- in order to reduce the number of features used -- I will select a single variable from each (or certain) set(s) to include in the model, rather than using PCA. 

#####Model Selection
The next step is to determine which model to use. I will consider the following methods (I've described results of each for reference): lda, qda, nb
- *Linear Discriminant Analysis (lda)*: accurary of 69.9%
- *Quadratic Discriminant Analysis (qda)*: accuracy of 89.3%
- *Naive Bayes (nb)*: this calculation exceeded 10+ minutes therefore I decided not to use it

The Quadratic Discriminant Analysis model (modelQDA) was the best balance of accuracy and efficiency/scalability, therefore I will move forward with this model. Note that I also trained this model with standardized variables (`preProcess=c("centered",scaled")`), but it did not outperform the non-transformed version of the model.
```
modelQDA <- train(classe~., data = trainingInput, method = "qda"); modelQDA

Quadratic Discriminant Analysis 

14718 samples
   52 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Bootstrapped (25 reps) 

Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ... 

Resampling results
  Accuracy   Kappa      Accuracy SD  Kappa SD   
  0.8948217  0.8670708  0.004669971  0.005868669
```

#####Variable Importance & Selection
The final step in my model selection process is to reduce the number of features to include in the model. To do this, I will use Variable Importance (`varImp()`) in the caret package. To select the final features, I have averaged the area under the curve (AUC) for the ROC calculation across each outcome variable (classe). 

```
modelQDAImp <- varImp(modelQDA, useModel = TRUE, scale = FALSE)
modelQDAImp
```

However, after attempting to reduce the number of features in the QDA model, I have found that the accuracy is reduced in all situations. I have also applied PCA pre-processing and standardization to this model, and it does now perform as well. Therefore I will move forward with the modelQDA specified above.


#####Cross Validation & Out of Sample Accuracy
As mentioned earlier, I omitted 25% of the data from "pml-training.csv" for testing. Using the steps below, I have calculated the **out-of-sample accuracy of 89.87%**.

```
predQDA <- predict(modelQDA, newdata = testing); predQDA
confusionMatrix(data = predQDA, testing$classe)

Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1301   45    0    1    0
         B   42  802   45    5   27
         C   20   91  803  104   40
         D   27    6    4  683   16
         E    5    5    3   11  818

Overall Statistics
               Accuracy : 0.8987         
                 95% CI : (0.8899, 0.907)
    No Information Rate : 0.2845         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.872          
 Mcnemar's Test P-Value : < 2.2e-16      

Statistics by Class:
                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9326   0.8451   0.9392   0.8495   0.9079
Specificity            0.9869   0.9699   0.9370   0.9871   0.9940
Pos Pred Value         0.9659   0.8708   0.7590   0.9280   0.9715
Neg Pred Value         0.9736   0.9631   0.9865   0.9710   0.9796
Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
Detection Rate         0.2653   0.1635   0.1637   0.1393   0.1668
Detection Prevalence   0.2747   0.1878   0.2157   0.1501   0.1717
Balanced Accuracy      0.9598   0.9075   0.9381   0.9183   0.9509
```



