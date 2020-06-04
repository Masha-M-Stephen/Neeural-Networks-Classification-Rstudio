---
title: "Nueral Networks"
author: "Masha"
date: "6/1/2020"
output:  
   md_document:
    variant: markdown_github
---

Description from UCI Machine Learning database:
The database consists of the multi-spectral values of pixels in 3x3 neighborhoods in a satellite image, and
the classification associated with the central pixel in each neighborhood. The aim is to predict this
classification, given the multi-spectral values. In the sample database, the class of a pixel is coded as a
number.

The Landsat satellite data is one of the many sources of information available for a scene. The
interpretation of a scene by integrating spatial data of diverse types and resolutions including
multispectral and radar data, maps indicating topography, land use etc. is expected to assume significant
importance with the onset of an era characterized by integrative approaches to remote sensing (for
example, NASA's Earth Observing System commencing this decade). Existing statistical methods are illequipped for handling such diverse data types. Note that this is not true for Landsat MSS data considered
in isolation (as in this sample database). This data satisfies the important requirements of being numerical
and at a single resolution, and standard maximum-likelihood classification performs very well.
Consequently, for this data, it should be interesting to compare the performance of other methods against
the statistical approach.

One frame of Landsat MSS imagery consists of four digital images of the same scene in different spectral
bands. Two of these are in the visible region (corresponding approximately to green and red regions of
the visible spectrum) and two are in the (near) infra-red. Each pixel is a 8-bit binary word, with 0
corresponding to black and 255 to white. The spatial resolution of a pixel is about 80m x 80m. Each
image contains 2340 x 3380 such pixels.

The database is a (tiny) sub-area of a scene, consisting of 82 x 100 pixels. Each line of data corresponds
to a 3x3 square neighborhood of pixels completely contained within the 82x100 sub-area. Each line
contains the pixel values in the four spectral bands (converted to ASCII) of each of the 9 pixels in the 3x3
neighborhood and a number indicating the classification label of the central pixel. The number is a code
for the following classes:
Number Class
1 red soil
2 cotton crop
3 grey soil
4 damp grey soil
5 soil with vegetation stubble
6 mixture class (all types present)
7 very damp grey soil
Note: There are no examples with class 6 in this dataset



```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```



```{r}
library(nnet)
```


```{r}
SATimage = read.csv("C:/Users/fb8502oa/Desktop/Github stuff/Neeural-Networks-Classification-Rstudio/SATimage (assg8).csv")

SATimage = data.frame(class=as.factor(SATimage$class),SATimage[,1:36])
```

```{r}
#Create a test and training set using the code below:
set.seed(888) 
testcases = sample(1:dim(SATimage)[1],1000,replace=F)
SATtest = SATimage[testcases,]
SATtrain = SATimage[-testcases,]
```

```{r}
#base neural network model
SAT.nn1 = nnet(class~., data = SATtrain, size = 9, decay = 0.025, maxit = 3000)
```

```{r}
#to see its summary
SAT.nn1
```

Misclassification FUNCTION
```{r}
misclass.nnet <- function(fit,y) {
temp <- table(predict(fit,type="class"),y)
cat("Table of Misclassification\n")
cat("(row = predicted, col = actual)\n")
print(temp)
cat("\n\n")
numcor <- sum(diag(temp))
numinc <- length(y) - numcor
mcr <- numinc/length(y)
cat(paste("Misclassification Rate = ",format(mcr,digits=3)))
cat("\n")
  }

```

Missclassification for the base model
```{r}
misclass.nnet(SAT.nn1, SATtrain$class)
```


SUMMARY:
After splitting the data into training and test set, I made a base model to see how the model did without
cross-validating it. The models had 9 sizes and a decay of 0.025. Its training set had a misclassification
rate of 0.0757.



CROSS-VALIDATION FUNCTION
```{r}
cnnet.cv = function (fit, y, data, B = 25, p = 0.667, size = 5, decay = 0.001, maxit = 5000,trace=T) 
{
    n <- length(y)
    cv <- rep(0, B)
    nin <- floor(n * p)
    out <- n - nin
    for (i in 1:B) {
        sam <- sample(1:n, nin)
        temp <- data[sam, ]
        fit2 <- nnet(formula(fit), data = temp, size = size, 
            decay = decay, maxit = maxit,trace=trace)
        ynew <- predict(fit2, newdata = data[-sam, ], type = "class")
        tab <- table(y[-sam], ynew)
        mc <- out - sum(diag(tab))
        cv[i] <- mc/out
    }
    cv
}

```


Using cross_validation
```{r}
SAT.cnnet1 = cnnet.cv(SAT.nn1, SATtrain$class, data = SATtrain, B=50, size = 9, decay =0.025, maxit = 4000)
summary(SAT.cnnet1)
```

SUMMARY:
I then wanted to see how the model would perform if it had been cross validated. SAT.cnnet1 had a B of
50, size of 9 and decay of 0.025. this model had 393 weights, a mean of 1643 and a max of 0.2290



LOOKING FOR A BETTER MODEL
```{r}
SAT.cnnet4 = cnnet.cv(SAT.nn1, SATtrain$class, data = SATtrain, B=50, size = 12, decay =0.050, maxit = 4000)
summary(SAT.cnnet4)
#lower than cnnet 1
```


```{r}
SAT.cnnet5 = cnnet.cv(SAT.nn1, SATtrain$class, data = SATtrain, B=50, size = 5, decay =0.025, maxit = 4000)
summary(SAT.cnnet5)
#lower than cnnet 1
```

SUMMARY:
In search for a better model, I fit numerous models in order to find the optimal size and decay. A lower
size such as 5 and decay of 0.025 had a mean of 0.1760 and a max of 0.6617. This model had a higher
misclassification rate compared to the first cross-validation model.



```{r}
SAT.cnnet6 = cnnet.cv(SAT.nn1, SATtrain$class, data = SATtrain, B=50, size = 15, decay =0.025, maxit = 4000)
summary(SAT.cnnet6)
#lower than cnnet 1
```

SUMMARY:
The next model that I considered had a much higher size. With a size of 8 and decay of 0.025, this model
had a mean of 0.1805 and a max of 0.2517. Even though the max was now lower, the mean was still
higher than the 0.16.



```{r}
SAT.cnnet8 = cnnet.cv(SAT.nn1, SATtrain$class, data = SATtrain, B=50, size = 8, decay =0.025, maxit = 3000)
summary(SAT.cnnet8)
#better
```
SUMMARY:
After trying many other models, I found an optimal size for the model. A size of 8 and decay of 0.025 had
350 weights, a mean of 0.1629 and a max of 0.188. This was the lowest misclassification rate on the
training data. 


FROM HERE WE TRY TO PREDICT THE TEST SET USING THE BEST MODEL FROM THE TRAINING SET MODELS.

```{r}
SAT.final = nnet(class~., data = SATtrain, size = 8,decay = 0.025, maxit= 4000)
misclass.nnet(SAT.final, SATtrain$class)
```

MISCLASS FUNCTION
```{r}
misclass = function(fit,y) {
  temp <- table(fit,y)
  cat("Table of Misclassification\n")
  cat("(row = predicted, col = actual)\n")
  print(temp)
  cat("\n\n")
  numcor <- sum(diag(temp))
  numinc <- length(y) - numcor
  mcr <- numinc/length(y)
  cat(paste("Misclassification Rate = ",format(mcr,digits=3)))
  cat("\n")
}

```


```{r}
yhat = predict(SAT.final, newdata = SATtest, type = "class")
misclass(yhat, SATtest$class)
```


```{r}
SAT.final2 = nnet(class~., data = SATtrain, size = 8,decay = 0.015, maxit= 3000)
misclass.nnet(SAT.final, SATtrain$class)
```

```{r}
yhat = predict(SAT.final2, newdata = SATtest, type = "class")
misclass(yhat, SATtest$class)
```

SUMMARY:
This model performed decently with the test set data. It had a size of 8, decay of 0.015 and a maxit of
4000. It had one of the lowest misclassification rates for the training set at 0.0789 and a 0.148
misclassification rate on the testing set.
The models with a higher or lower than size 8 had a larger misclassification rate compared to my final
model.
