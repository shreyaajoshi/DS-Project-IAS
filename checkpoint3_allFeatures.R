#setting the working directory
setwd('/Users/Shreya/Documents/KDD/Project')

#import the required libraries
library(R.matlab)
library(party)
library(partykit)
library(randomForest)
library(leaps)

#read the dataset
rawData = readMat('impact_database(8factors).mat')

#classifying feature
iasData=rawData$IASScore

#categorize the continuous values
impactColumn<-c()
#width<-c()
for (index in iasData){
  if (index>50){
    impact=1
    impactColumn=c(impactColumn,impact)
  }
  else{
    impact=0
    impactColumn=c(impactColumn,impact)
  }
}

#finding the maximum linear accleration
linAccl_list<-c()
count=nrow(rawData$MagLinAccel)
width<-c()
for (index in 1:count){
  #print(index)
  max_again=max(rawData$MagLinAccel[index,])
  linAccl_list=c(linAccl_list,max_again)
  
  indMax=which.max(rawData$MagLinAccel[index,])
  halfmaxlin=max_again/2
  firstHalf<-c(rawData$MagLinAccel[index,1:indMax])
  firstHalfRev=rev(firstHalf)
  needed_value1=0
  for(jindex in firstHalfRev) {
    if (jindex<halfmaxlin){
      needed_value1=jindex
      break
    }
  }
  if (needed_value1==0){
    needed_value1=firstHalf[1]
  }
  firstHalfValInd=which(rawData$MagLinAccel[index,]==needed_value1)
  if (length(firstHalfValInd)>=2){
    firstHalfValInd=firstHalfValInd[2]
  }
  
  secondHalf<-c(rawData$MagLinAccel[index,indMax:80])
  
  needed_value=0
  for(kindex in secondHalf){
    if (kindex<halfmaxlin){
      needed_value=kindex
      break
    }
  }
  if (needed_value==0){
    needed_value=secondHalf[length(secondHalf)]
  }
  
  secondHalfValInd=which(rawData$MagLinAccel[index,]==needed_value)
  if (length(secondHalfValInd)>=2){
    secondHalfValInd=secondHalfValInd[1]
  }
  widthAppend=abs((secondHalfValInd-firstHalfValInd))+1
  #print(widthAppend)
  width=c(width,widthAppend)
  #print(index)
}

#rawData$Impact_Column<-impactColumn
#rawData$Width<-width

#finaing maximum rotational velocity
rotVel_list<-c()
for (index in 1:count){
  max_again_rot=max(rawData$MagRotVel[index,])
  rotVel_list=c(rotVel_list,max_again_rot)
}
rotAccl_list<-c()
for (index in 1:count){
  rot_accl_val=max(rawData$MagRotAccel[index,])
  rotAccl_list=c(rotAccl_list,rot_accl_val)
}

#Understanding data using the summary

regfit.full=regsubsets(rawData$impact~rawData$HIC15+rawData$RIC15+rawData$PeakLinAccl+rawData$PeakRotVel+rawData$PeakRotAccl+rawData$width+rawData$gambit.score,rawData)
reg.summary=summary(regfit.full)
reg.summary
reg.summary$rsq

par ( mfrow = c (2 ,2) )
plot ( reg.summary$rss , xlab ="Number of Variables", ylab ="RSS",type ="l")
plot ( reg.summary$adjr2 , xlab ="Number of Variables",ylab ="Adjusted RSq",type ="l")
which.max(reg.summary$adjr2)

points(7,reg.summary$adjr2[7], col="red", cex=2, pch=20)
plot(reg.summary$cp,xlab="# Variables", ylab="Cp",type="l")
which.min(reg.summary$cp)
points(7,reg.summary$cp[7], col="red", cex=2, pch=20)


which.min(reg.summary$bic)
plot(reg.summary$bic,xlab="# Variables", ylab="BIC",type="l")
points(5,reg.summary$bic[5], col="red", cex=2, pch=20)

par(mfrow=c(1,1))
plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp")
plot(regfit.full,scale="bic")

coef(regfit.full)

#stepwise
regfit.fwd=regsubsets(rawData$impact~rawData$HIC15+rawData$RIC15+rawData$PeakLinAccl+rawData$PeakRotVel+rawData$PeakRotAccl+rawData$width+rawData$gambit.score,data=rawData,method="forward")
summary(regfit.fwd)
regfit.bwd=regsubsets(rawData$impact~rawData$HIC15+rawData$RIC15+rawData$PeakLinAccl+rawData$PeakRotVel+rawData$PeakRotAccl+rawData$width+rawData$gambit.score,data=rawData,method="backward")
summary(regfit.bwd)


#creating DataFrame for classifiers

hic15=rawData$HIC15
ric15=rawData$RIC15
gambit_score=rawData$gambit.score
# linAccl_list=rawData$PeakLinAccl
# rotVel_list=rawData$PeakRotVel
# rotAccl_list=rawData$PeakRotAccl
# impactColumn=rawData$impact

rawData=data.frame(rawData,impactColumn,linAccl_list,rotVel_list,rotAccl_list,hic15,ric15,width,gambit_score)
set.seed (2)
train = sample(1:nrow(rawData),1480)
rawData.test = rawData [-train,]
impactColumn.test = impactColumn [ - train ]

# Create the random forest with all predictors
output.forest <- randomForest(factor(impactColumn) ~ hic15+linAccl_list+rotVel_list+rotAccl_list+ric15+width+gambit_score , data=rawData,importance=TRUE)

output.forest = randomForest(factor(impactColumn)~hic15+linAccl_list + rotVel_list+rotAccl_list+ric15+width+gambit_score,rawData , subset=train )
output.forest.pred = predict (output.forest,rawData.test, type ="class")
table ( output.forest.pred , impactColumn.test)

# View the forest results.
print(output.forest)
plot(output.forest)
# Importance of each predictor.
print(importance(output.forest))
varImpPlot(output.forest)

attach(rawData)


x=model.matrix(impact ~ HIC15+RIC15+width+PeakLinAccl+PeakRotAccl+PeakRotVel+gambit.score,rawData)[,-1]
y=rawData$impact

#training and test data
set.seed(1)
train=sample(1:nrow(x),nrow(x)/2)
test=(-train)
y.test=y[test]

# ridge regression
grid=10^seq(10,-2,length=100)
grid

x=model.matrix(impact~HIC15+PeakLinAccl+PeakRotAccl+PeakRotVel+RIC15+width+gambit.score, data=rawData)[,-1]
y=rawData$impact

ridge.mod=glmnet(x,y,alpha = 0,lambda = grid)
dim(coef(ridge.mod))

ridge.mod$lambda[50] # When lambda = 11498
coef(ridge.mod)[,50]
sqrt(sum(coef(ridge.mod)[-1,50]^2))

ridge.mod$lambda[60] # When lambda = 705 
coef(ridge.mod)[,60]
sqrt(sum(coef(ridge.mod)[-1,60]^2))

predict(ridge.mod, s=50, type="coefficients")[1:8,]

# Splitting the data into test and train
set.seed(1)
train=sample(1:nrow(x),nrow(x)/2)
test=(-train)
y.test=y[test]

ridge.mod=glmnet(x[train,], y[train], alpha = 0,lambda = grid, thresh = 1e-12)
ridge.pred=predict(ridge.mod,s=4,newx = x[test,]) # for small value of lambda
mean((ridge.pred-y.test)^2)

ridge.pred=predict(ridge.mod,s=1e10,newx = x[test,]) # for big value of lambda
mean((ridge.pred-y.test)^2)

ridge.pred=predict(ridge.mod,s=0,newx = x[test,], exact = T)
mean((ridge.pred-y.test)^2)
lm(y~x, subset = train)
predict(ridge.mod,s=0,exact = T, type="coefficients")[1:8,]

# Choosing tuning parameter lambda using Cross Validation
set.seed(1)
cv.out=cv.glmnet(x[train,],y[train],alpha=0)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam # Optimal value of lambda
ridge.pred=predict(ridge.mod,s=bestlam, newx = x[test,])
mean((ridge.pred-y.test)^2)
out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlam)[1:8,]

##Lasso
set.seed(1)
cv.out=cv.glmnet(x[train,], y[train], alpha=1)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam
lasso.pred=predict(lasso.mod,s=bestlam,newx=x[test,])
mean((lasso.pred-y.test)^2)

out=glmnet(x,y,alpha=1, lambda=grid)
lasso.coef=predict(out,type="coefficients", s=bestlam)[1:8,]
lasso.coef
lasso.coef[lasso.coef !=0]

#random forest after Lasso
output.forest_lasso <- randomForest(factor(impactColumn) ~ hic15+linAccl_list+rotVel_list+rotAccl_list+ric15+width, data=rawData,importance=TRUE)
output.forest_lasso = randomForest(factor(impactColumn)~hic15+linAccl_list + rotVel_list+rotAccl_list+ric15+width,rawData , subset=train )
output.forest.pred_lasso = predict (output.forest,rawData.test, type ="class")
table ( output.forest.pred_lasso , impactColumn.test)

#logistic regression
linearNew = glm(factor(impactColumn) ~ hic15+linAccl_list + rotVel_list+rotAccl_list+ric15+width+gambit_score, family = binomial(link = "logit"),data = rawData)
summary(linearNew)

linearNew=glm(factor(impactColumn) ~ hic15+linAccl_list + rotVel_list+rotAccl_list+ric15+width+gambit_score, family = binomial(link = "logit"),rawData,subset = train)
linearNew.pred=predict(linearNew,rawData.test, type ="response")

glm.pred = rep (0 ,495)
glm.pred [ linearNew.pred >.5]=1
table ( glm.pred , impactColumn.test)


#decision tree
library(tree)
tree.fitNew1=tree(factor(impactColumn)~hic15+linAccl_list + rotVel_list+rotAccl_list+ric15+width+gambit_score , data=rawData)
summary(tree.fitNew1)
plot (tree.fitNew1)
text (tree.fitNew1, pretty =0)

tree.fitNew1 = tree(factor(impactColumn)~hic15+linAccl_list + rotVel_list+rotAccl_list+ric15+width+gambit_score , subset=train )
tree.pred = predict(tree.fitNew1,rawData.test, type ="class")
table ( tree.pred , impactColumn.test)


tree.fitNew1.cv <- cv.tree(tree.fitNew1)
summary(tree.fitNew1.cv)
plot(tree.fitNew1.cv)

#svm
install.packages('e1071')
library ( e1071 )
svmfit = svm (factor(impactColumn)~hic15+linAccl_list + rotVel_list+rotAccl_list+ric15+width+gambit_score,data=rawData,scale = FALSE )
summary(svmfit)

svmfit = svm (factor(impactColumn)~hic15+linAccl_list + rotVel_list+rotAccl_list+ric15+width+gambit_score,rawData,subset = train )
svmfit.pred = predict (svmfit,rawData.test, type ="class")
table ( svmfit.pred , impactColumn.test)


#lda
library(MASS)
ldafit = lda(factor(impactColumn)~hic15+linAccl_list + rotVel_list+rotAccl_list+ric15+width+gambit_score,data=rawData)
summary(ldafit)

ldafit = lda(factor(impactColumn)~hic15+linAccl_list + rotVel_list+rotAccl_list+ric15+width+gambit_score,rawData,subset = train )
ldafit.pred = predict (ldafit,rawData.test)
names(ldafit.pred)
table ( ldafit.pred$class , impactColumn.test)

#qda
qdafit = qda(factor(impactColumn)~hic15+linAccl_list + rotVel_list+rotAccl_list+ric15+width+gambit_score,data=rawData)
summary(qdafit)

qdafit = qda(factor(impactColumn)~hic15+linAccl_list + rotVel_list+rotAccl_list+ric15+width+gambit_score,rawData,subset = train )
qdafit.pred = predict (qdafit,rawData.test)
names(qdafit.pred)
table ( qdafit.pred$class , impactColumn.test)


