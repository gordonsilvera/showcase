

# import origninal datasets
mainTrain <- read.table("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", header = TRUE, sep = ",")
mainTest <- read.table("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", header = TRUE, sep = ",")

# subset to only complete data
mainTrain1 <- subset(mainTrain, select=c("user_name",	"raw_timestamp_part_1",	"raw_timestamp_part_2",	"cvtd_timestamp",	"new_window",	"num_window",	
                                          "roll_belt",	"pitch_belt",	"yaw_belt",	"total_accel_belt",	"gyros_belt_x",	"gyros_belt_y",	"gyros_belt_z",	
                                          "accel_belt_x",	"accel_belt_y",	"accel_belt_z",	"magnet_belt_x",	"magnet_belt_y",	"magnet_belt_z",	
                                          "roll_arm",	"pitch_arm",	"yaw_arm",	"total_accel_arm",	"gyros_arm_x",	"gyros_arm_y",	"gyros_arm_z",	
                                          "accel_arm_x",	"accel_arm_y",	"accel_arm_z",	"magnet_arm_x",	"magnet_arm_y",	"magnet_arm_z",	"roll_dumbbell",	
                                          "pitch_dumbbell",	"yaw_dumbbell",	"total_accel_dumbbell",	"gyros_dumbbell_x",	"gyros_dumbbell_y",	"gyros_dumbbell_z",	
                                          "accel_dumbbell_x",	"accel_dumbbell_y",	"accel_dumbbell_z",	"magnet_dumbbell_x",	"magnet_dumbbell_y",	
                                          "magnet_dumbbell_z",	"roll_forearm",	"pitch_forearm",	"yaw_forearm",	"total_accel_forearm",	
                                          "gyros_forearm_x",	"gyros_forearm_y",	"gyros_forearm_z",	"accel_forearm_x",	"accel_forearm_y",	
                                          "accel_forearm_z",	"magnet_forearm_x",	"magnet_forearm_y",	"magnet_forearm_z",	"classe"))

# import libraries
library(caret); library(kernlab); library(MASS); library(klaR)


# create MODEL training dataset
inTrain <- createDataPartition(y=mainTrain1$classe, p=0.75, list=FALSE)
training <- mainTrain1[inTrain,]
testing <- mainTrain1[-inTrain,]
dim(training); dim(testing)


# EXPLORATORY DATA ANALYSIS
summary(mainTrain1)
qplot(pitch_dumbbell, total_accel_dumbbell, colour=classe, data=training)   # sample Q-plot
qplot(total_accel_forearm, colour=classe, data=training, geom="density")    # sample Density Plots


# principle components exploration
M <- abs(cor(training[,7:58]))
diag(M) <- 0
which(M > 0.8, arr.ind = T)



# training models (LDA, w/ PCA)
modelFit1 <- train(classe~., data = trainingInput, method = "lda", preProcess=c("pca")); modelFit1  # Accuracy = 0.5199
predFit1 <- predict(modelFit1, trainingInput)
table(predFit1, trainingInput$classe)


# training models (LDA, no PCA)
modelFit2 <- train(classe~., data = trainingInput, method = "lda"); modelFit2  # Accuracy = 0.6989
predFit2 <- predict(modelFit2, trainingInput)
table(predFit2, trainingInput$classe)


# training models (QDA)
modelQDA <- train(classe~., data = trainingInput, method = "qda"); modelQDA

modelQDA <- varImp(modelFit3, scale=FALSE); modelQDA



# cross validation (QDA)
predQDA <- predict(modelQDA, newdata = testing); predQDA
confusionMatrix(data = predQDA, testing$classe)




