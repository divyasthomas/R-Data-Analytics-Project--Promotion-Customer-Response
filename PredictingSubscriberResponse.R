
# Load required R packages
library("dplyr")
library("car")
library("forcats")
library("rpart")
library("rpart.plot")
library("nnet")
library("foreign")
library("corrplot")
library("randomForest")
library("pdp")
library("ggplot2")
library(nnet)
library(caret)
library(e1071)
library(effects)
source('BCA_functions_source_file.R')

##***********************************<<  LOADING DATASET >> ***************************************
#Loading file
QK2=read.csv("QK.csv")

##***********************************<<  ExPLORING DATASET >> ***************************************
#Looking at data
summary(QK2$SUBSCRIBE) #The classifier might be biased.....In this case towards N...
gglimpse(QK2)
variable.summary(QK2)

##***********************************<<  Data Cleaning>> ***************************************

  
# Not useful statistically since they are arbritary numbers
row.names(QK2) <-QK2$custid # Set "ID" as record name
QK2$custid <-NULL

# This function, in forcats package, recodes factor (categorical) missing values
# and keeps variable type as factor. See ?forcats::fct_explicit_na
QK2$Disc <- fct_explicit_na(QK2$Disc, # Factor of interest
                            na_level ="ND") # replacement value

#deleting postal code(not meanngful)
QK2$Pcode <-NULL

#deleting Title: "Mr"  "Ms"  "Mrs"  or "Dr" (35% missing)
QK2$Title <-NULL

#deleting Weeks3Meals since it is a trivially related predictor
QK2$Weeks3Meals <-NULL

#DELETING x
QK2$X<-NULL

#LastOrder is date, and not factor
currentdate="2018/03/05" #date of sending prommotion offer
QK2$LastOrder = as.numeric(as.Date(currentdate)-as.Date(QK2$LastOrder, format="%Y-%m-%d"))
range(QK2$LastOrder)

#small % of missing values remaining in the geodemographic variables - a maximum of 3.4%.
# Remove all missing values from those variables
QK3 <- QK2[!is.na(QK2$DA_Single),]
QK3 <- QK2[!is.na(QK2$DA_Income),]
QK3 <- QK2[!is.na(QK2$DA_Over60),]

#looking at the data summary
variable.summary(QK3)


##***********************************<<  MODELLING>> ***************************************


# Copy & paste given variable names into the predictor list
paste(names(QK3),collapse =" + ")

##***************** << RANDOM FOREST >>************************

# ********all variables in random forest***************
QKForestAllv <- randomForest(formula =SUBSCRIBE  ~ Disc + LastOrder + DA_Income + DA_Under20 + DA_Over60 + DA_Single + NumDeliv + NumMeals + MealsPerDeliv + Healthy + Veggie + Meaty + Special + TotPurch, data = filter(QK3, Sample =="Estimation"),
                             importance =TRUE,
                             ntree =500,mtry =14)
# Contingency Table
QKForestAllv[["confusion"]]

#******** all variables in final logistic step into random forest*********
QKForest_step <- randomForest(formula =SUBSCRIBE  ~  DA_Income + DA_Under20 + NumDeliv + 
                                MealsPerDeliv + Veggie + LastOrder+ TotPurch, data = filter(QK3, Sample =="Estimation"),
                              importance =TRUE,
                              ntree =500,mtry =7)

# Contingency Table
QKForest_step[["confusion"]]

# ********all variables in final logistic step into random forest+disc**************************
QKForest_step2 <- randomForest(formula =SUBSCRIBE  ~ Disc+ DA_Income + DA_Under20 + NumDeliv + 
                                MealsPerDeliv + Veggie + LastOrder+ TotPurch, data = filter(QK3, Sample =="Estimation"),
                              importance =TRUE,
                              ntree =500,mtry =7)
# Contingency Table
QKForest_step2[["confusion"]]

 
#Experiment with weighting to penalize for imbalanced data
#this helps with false negative error rate, but increases OOB estimate error rate
forest_weighted = randomForest(SUBSCRIBE ~ Disc + LastOrder + DA_Income + DA_Under20 + DA_Over60 + 
                         DA_Single + NumDeliv + NumMeals + MealsPerDeliv + Healthy + Veggie + 
                         Meaty + Special + TotPurch,
                       data = filter(QK3, Sample =="Estimation"),
                       classwt = c(0.0005, 1000),
                       importance=TRUE,
                       ntree=500, mytry=14)
#*********************************<<PDP PLOTS>>***********************************************
# Variable importance
# Variable importance
varImpPlot(QKForestAllv,type =2,
           main="QKForestAllv", # title
           cex =0.7) # font size

# Variable importance
# Variable importance
varImpPlot(QKForest_step ,type =2,
           main="QKForestAllv", # title
           cex =0.7) # font size
#checking base case 
levels(QK3$SUBSCRIBE)           

# partial dependence plots
#  NumDeliv 
partial(QKForestAllv,pred.var ="NumDeliv", # target and predictor
        prob =TRUE, # probabilities on yaxis 
        which.class =2, # predict level 2, "Y"
        plot =TRUE, # generate plot
        rug =TRUE, # plot decile hashmarks
        plot.engine ="ggplot2")

# TotPurch 
partial(QKForestAllv,pred.var ="TotPurch",
        prob =TRUE, # probabilities on yaxis
        which.class =2, # predict level 2, "Y"
        plot =TRUE, # generate plot
        rug =TRUE, # plot decile hashmarks
        plot.engine ="ggplot2")

# "NumMeals"
partial(QKForestAllv,pred.var ="NumMeals",
        prob =TRUE, # probabilities on yaxis
        which.class =2, # predict level 2, "Y"
        plot =TRUE, # generate plot
        rug =TRUE, # plot decile hashmarks
        plot.engine ="ggplot2")
#"LastOrder"
partial(QKForestAllv,pred.var ="LastOrder", 
        prob =TRUE, # probabilities on yaxis 
        which.class =2, # predict level 2, "Y"
        plot =TRUE, # generate plot
        rug =TRUE, # plot decile hashmarks
        plot.engine ="ggplot2")


#"MealsPerDeliv"
partial(QKForestAllv,pred.var ="MealsPerDeliv", 
        prob =TRUE, # probabilities on yaxis 
        which.class =2, # predict level 2, "Y"
        plot =TRUE, # generate plot
        rug =TRUE, # plot decile hashmarks
        plot.engine ="ggplot2")


#"DA_Under20"
partial(QKForestAllv,pred.var ="DA_Under20", 
        prob =TRUE, # probabilities on yaxis 
        which.class =2, # predict level 2, "Y"
        plot =TRUE, # generate plot
        rug =TRUE, # plot decile hashmarks
        plot.engine ="ggplot2")

# Trim the top 10%
# NumDeliv
QKForestAllv.trim <- partial(QKForestAllv,pred.var ="NumDeliv",
                             prob =TRUE,
                             which.class =2,
                             quantiles =TRUE, # prepare data trimming
                             probs = seq(from =0.0,to =0.9,by =0.02), # of bottom 90%
                             plot=FALSE) # generate data, no plot
plotPartial(QKForestAllv.trim, # and pass data to plotting function
            rug =TRUE,
            train = filter(QK3, Sample == "Estimation"))

# TotPurch 
QKForestAllv.trim <- partial(QKForestAllv,pred.var ="TotPurch",
                             prob =TRUE,
                             which.class =2,
                             quantiles =TRUE,
                             probs = seq(from =0.0,to =0.9,by =0.02),
                             plot=FALSE)
plotPartial(QKForestAllv.trim,
            rug =TRUE,
            train = filter(QK3, Sample == "Estimation"))


# NumMeals
QKForestAllv.trim <- partial(QKForestAllv,pred.var ="NumMeals",
                             prob =TRUE,
                             which.class =2,
                             quantiles =TRUE,
                             probs = seq(from =0.0,to =0.9,by =0.02),
                             plot=FALSE)
plotPartial(QKForestAllv.trim,
            rug =TRUE,
            train = filter(QK3, Sample == "Estimation"))

# LastOrder
QKForestAllv.trim <- partial(QKForestAllv,pred.var ="LastOrder",
                             prob =TRUE,
                             which.class =2,
                             quantiles =TRUE,
                             probs = seq(from =0.0,to =0.9,by =0.02),
                             plot=FALSE)
plotPartial(QKForestAllv.trim,
            rug =TRUE,
            train = filter(QK3, Sample == "Estimation"))

#examin the PDPs for THE ONLY  categorical predictor DISC 
partial(QKForestAllv,pred.var ="Disc",
        which.class =2,
        plot =TRUE,
        rug =TRUE,plot.engine ="ggplot2",
        prob =TRUE)

#**************************<< VARIABLE TRANSFORMATIONS>>*********************

#Since ND has lowest(dept)- use it to set base level in dummy variable in logistic regression
#set the base levels
levels(QK3$Disc)
QK3$Disc <- relevel(QK3$Disc,"ND")

levels(QK3$Disc)

summary(select(QK3,LastOrder,TotPurch,MealsPerDeliv))
QK3$log_lastorder <- log(QK3$LastOrder)
QK3$log_TotPurch <- log(QK3$TotPurch)
QK3$log_MealsPerDeliv <-log(QK3$MealsPerDeliv)


#**************************<< MULTICOLLINEARITY PLOTS>>*********************
#finding multicollinearity in dataset

# Correlation Matrix
# Select numeric columns only, then calculate and print correlation coefficients
corrMatrix <- cor(select_if(QK3, is.numeric)) # see ?dplyr::select_if
# temporarily reduce the number of output digits for easier inspection
options(digits =2)
corrMatrix
options(digits =7) # then reset output digits

# Visualize correlation
corrplot(corrMatrix,method="number",type="lower",
         diag =FALSE,number.cex =0.7)


#creating validation and estimation sets
validation=filter(QK3, Sample == "Validation")
Estimation=filter(QK3, Sample == "Estimation")

##***************** << BASIC LOGISTIC MODEL >>************************
# Create a logistic regression model
QKLogis <- glm(formula =SUBSCRIBE ~ Disc + LastOrder+ DA_Income + DA_Under20 + DA_Over60 + DA_Single + NumDeliv + NumMeals + MealsPerDeliv + Healthy + Veggie + Meaty + Special + TotPurch,
               data = filter(QK3, Sample =="Estimation"),
               family = binomial(logit))
# Print
summary(QKLogis)

# Calculate and print McFadden R square (See Logistic Regression Chapter)
MR2 <-1 - (QKLogis$deviance / QKLogis$null.deviance)
MR2.3<- round(MR2,digits =3)
print(paste("McFadden Rsquared: ",MR2.3))# McFadden Rsquared:  0.183 AIC: 596.87


##***************** << ADJUSTED LOGISTIC MODEL >>************************
adjust_lr <- glm(formula = SUBSCRIBE ~ DA_Income + DA_Under20 + DA_Over60 + DA_Single + NumDeliv+ MealsPerDeliv + Healthy + Veggie + Meaty + log_TotPurch + Special + log_lastorder, 
                 data = filter(QK3, Sample =="Estimation"), 
                 family = binomial(logit)) 
# Print
summary(adjust_lr)

# Calculate and print McFadden R square (See Logistic Regression Chapter)
MR2 <-1 - (adjust_lr$deviance / adjust_lr$null.deviance)
MR2.3<- round(MR2,digits =3)
print(paste("McFadden Rsquared: ",MR2.3))# McFadden Rsquared:  0.177 AIC: 532.88

##***************** << STEPWISE LOGISTIC MODEL >>************************
#Run a stepwise regression using the "Weslogis" model
QKStep <- step(QKLogis,direction ="both")
summary(QKStep)

# McFadden R2
MR2.step <-1 - (QKStep$deviance / QKStep$null.deviance)
MR2.step.3<- round(MR2.step,digits =3)
print(paste("McFadden Rsquared: ",MR2.step.3)) # "McFadden Rsquared:  0.18" AIC: 588.6

##***************** << ADJUSTED STEPWISE LOGISTIC MODEL >>************************
#Run a stepwise regression using the "Weslogis" model
adjusted_QKStep <- step(adjust_lr,direction ="both")
summary(adjusted_QKStep)

# McFadden R2
MR2.step <-1 - (adjusted_QKStep$deviance / adjusted_QKStep$null.deviance)
MR2.step.3<- round(MR2.step,digits =3)
print(paste("McFadden Rsquared: ",MR2.step.3)) # "McFadden Rsquared:  0.271" AIC: 521.34

plot(allEffects(adjusted_QKStep),type="response")

##***************** << ADJUSTED STEPWISE LOGISTIC MODEL2 >>************************
adjust_lr2 <- glm(formula = SUBSCRIBE ~  DA_Income+ DA_Under20 +  NumDeliv+ log_MealsPerDeliv +  Veggie +  log_TotPurch +  log_lastorder, 
                 data = filter(QK3, Sample =="Estimation"), 
                 family = binomial(logit)) 
# Print
summary(adjust_lr2)

# Calculate and print McFadden R square (See Logistic Regression Chapter)
MR2 <-1 - (adjust_lr2$deviance / adjust_lr2$null.deviance)
MR2.3<- round(MR2,digits =3)
print(paste("McFadden Rsquared: ",MR2.3))# McFadden Rsquared:  0.281 AIC: 512.93

#Run a stepwise regression using the "Weslogis" model
adjusted_QKStep2 <- step(adjust_lr2,direction ="both")
summary(adjusted_QKStep2)

# McFadden R2
MR2.step <-1 - (adjusted_QKStep2$deviance / adjusted_QKStep2$null.deviance)
MR2.step.3<- round(MR2.step,digits =3)
print(paste("McFadden Rsquared: ",MR2.step.3)) # "McFadden Rsquared:  0.271" AIC: 521.34

##***************** << NEURAL MODEL WITH LOGISTIC STEP variables  >>************************
# 4 node nnet1
QKNnetStep1 <- Nnet(formula =SUBSCRIBE ~ Disc + DA_Income + DA_Under20 + NumDeliv + 
                  NumMeals + MealsPerDeliv + Veggie + LastOrder,
                data = filter(QK3, Sample =="Estimation"),decay =0.15,size =4)

# 4 node nnet2
QKNnetStep2 <- Nnet(formula =SUBSCRIBE ~ DA_Income + DA_Under20 + NumDeliv + 
                       MealsPerDeliv + Veggie + LastOrder+ TotPurch,
                    data = filter(QK3, Sample =="Estimation"),decay =0.15,size =4)

# 4 node nnet3
QKNnetStep3 <- Nnet(formula =SUBSCRIBE ~ DA_Income + DA_Under20 + NumDeliv + 
                      NumMeals+ Veggie + LastOrder+ TotPurch,
                    data = filter(QK3, Sample =="Estimation"),decay =0.15,size =4)
# 4 node nnet4
QKNnetStep4 <- Nnet(formula =SUBSCRIBE ~ Disc+DA_Income + DA_Under20 + NumDeliv + 
                      NumMeals+ Veggie + LastOrder+ TotPurch,
                    data = filter(QK3, Sample =="Estimation"),decay =0.15,size =4)


##***************** << NEURAL MODEL WITH all variables  >>************************
## All variables in nnet
QKNetAllv <- Nnet(formula =SUBSCRIBE ~ Disc +  DA_Income + DA_Under20 + DA_Over60 + DA_Single + NumDeliv + NumMeals + MealsPerDeliv + Healthy + Veggie + Meaty + Special + TotPurch + LastOrder,
                  data = filter(QK3, Sample =="Estimation"),decay =0.15,size =4)

##***************** << COMPARING MODELS using validation DS >>************************


#FOREST CONFUSION MATRIX
# all variables in random forest
QKForestAllv_val <- randomForest(formula =SUBSCRIBE  ~ Disc +  DA_Income + DA_Under20 + DA_Over60 + DA_Single + NumDeliv + NumMeals + MealsPerDeliv + Healthy + Veggie + Meaty + Special + TotPurch + LastOrder, data = validation,
                             importance =TRUE,
                             ntree =500,mtry =12)

confusionMatrix(QKForestAllv_val$predicted ,validation$SUBSCRIBE)


# all variables in random forest
QKForest_step_val <- randomForest(formula =SUBSCRIBE  ~ Disc+ DA_Income + DA_Under20 + NumDeliv + 
                                    MealsPerDeliv + Veggie + LastOrder+ TotPurch, data = validation,
                                 importance =TRUE,
                                 ntree =500,mtry =12)

confusionMatrix(QKForest_step_val$predicted ,validation$SUBSCRIBE)


#NEURAL LOGSTEP CONFUSION MATRIX
# 4 node nnet
QKNnetStep_val1=ifelse(predict(QKNnetStep1,newdata = validation,type="raw")>0.5,"Y","N")

confusionMatrix(as.factor(QKNnetStep_val1) ,validation$SUBSCRIBE)

# 4 node nnet
QKNnetStep_val2=ifelse(predict(QKNnetStep2,newdata = validation,type="raw")>0.5,"Y","N")

confusionMatrix(as.factor(QKNnetStep_val2) ,validation$SUBSCRIBE)

QKNnetStep_val3=ifelse(predict(QKNnetStep3,newdata = validation,type="raw")>0.5,"Y","N")

confusionMatrix(as.factor(QKNnetStep_val3) ,validation$SUBSCRIBE)

QKNnetStep_val4=ifelse(predict(QKNnetStep4,newdata = validation,type="raw")>0.5,"Y","N")

confusionMatrix(as.factor(QKNnetStep_val4) ,validation$SUBSCRIBE)

#NEURAL ALL CONFUSION MATRIX
QKNetAllv_val1=ifelse(predict(QKNetAllv,newdata = validation,type="raw")>0.5,"Y","N")

confusionMatrix(as.factor(QKNetAllv_val1) ,validation$SUBSCRIBE)

#LOGISTIC STEP CONFUSION MATRIX
QKStep_val1=ifelse(predict(QKStep,newdata = validation,type="response")>0.5,"Y","N")

confusionMatrix(as.factor(QKStep_val1) ,validation$SUBSCRIBE)

#adjusted LOGISTIC
adjusted_QKStep_val1=ifelse(predict(adjusted_QKStep,newdata = validation,type="response")>0.5,"Y","N")

confusionMatrix(as.factor(adjusted_QKStep_val1) ,validation$SUBSCRIBE)

#BASIC LOGISTIC
adjusted_QKStep_val2=ifelse(predict(adjusted_QKStep2,newdata = validation,type="response")>0.5,"Y","N")

confusionMatrix(as.factor(adjusted_QKStep_val2) ,validation$SUBSCRIBE)

##***************** << LIFT CHARTS >>************************


# Compare on Validation Sample -forests
lift.chart(modelList = c("adjusted_QKStep","QKStep", "QKForest_step2","forest_weighted","QKForest_step","QKForestAllv"),
           data = filter(QK3, Sample == "Validation"),
           targLevel ="Y",trueResp =0.165,
           type ="cumulative",sub ="Validation")

# Compare on Validation Sample -nnet
lift.chart(modelList = c("QKNnetStep1","QKNnetStep4","QKNnetStep3","QKNnetStep2"),
           data = filter(QK3, Sample == "Validation"),
           targLevel ="Y",trueResp =0.165,
           type ="cumulative",sub ="Validation")


# Compare on Validation Sample - Final lift chart
lift.chart(modelList = c("adjusted_QKStep","QKStep", "QKNnetStep3","QKNetAllv","QKForest_step","QKForestAllv","forest_weighted"),
           data = filter(QK3, Sample == "Validation"),
           targLevel ="Y",trueResp =0.165,
           type ="cumulative",sub ="Validation")


# OF THE LOT forest step seems to be best at prediction-
##***************** << CALCULATING PREDICTED TARGET VALUES >>************************
# Raw Estimated Probabilities added to data in ScoreRaw
QK3$ScoreRaw <- rawProbScore(model ="QKNnetStep3 ",
                             data =QK3,
                             targLevel ="Y")

#Rank Order - rank individuals in dataframe from best to worst, in ScoreRank
QK3$ScoreRAnk <- rankScore(model ="QKNnetStep3 ",
                           data =QK3,
                           targLevel ="Y")

##***************** << WRITING DATA TO FILE >>************************

# Put customer ID's back as a variable
QK3$custid  <- rownames(QK3)

#WRITING FILE
write.csv(QK3, "modelled_data.csv")
