# Deep-Survival-Analysis-for-Remaining-Process-Runtime-Prediction

## Introduction
This study delves into the crucial task of estimating process completion times, which holds significant practical implications for decision-making, efficiency enhancement, cost reduction, and resource optimization. Specifically, the research focuses on predicting remaining process runtimes in business processes, leveraging deep learning-based survival analysis, notably the DeepSurv framework. Real-life event log data from loan applications forms the basis of the analysis. By comparing the performance of the DeepSurv model with established methods, the study underscores both its strengths and limitations. Additionally, we discuss potential reasons behind the observed results, contributing to a deeper understanding of the predictive capabilities of deep learning-based survival analysis in business process management.

## Dataset
We conducted the experiment using the real-life dataset which is publicly available, on the prediction of the remaining duration of loan application processes. The historical data is from a Dutch Financial Institute, containing all applications filed trough an online system in 2016 and their subsequent events until February 1st, 2017, which is available at the 4TU Center for Research Data[1]. The data is in the form of an event log, which consists of many cases, i.e. the instances of the business process. 
<div align=center>
<img width='600' src='https://github.com/Shu-Shine/Deep-Survival-Analysis-for-Remaining-Process-Runtime-Prediction/blob/main/images/Loan_applications_process.png'/>  
Fig. 1 Loan applications process[2].
<img width='500' src='https://github.com/Shu-Shine/Deep-Survival-Analysis-for-Remaining-Process-Runtime-Prediction/blob/main/images/event_log.png'/> 
Fig. 2 Event log data.
</div>

## Method
### Feature Engineering
*Firstly we introduced the event indicator, taking “Application Pending” as our event of interest.
*Then for the static case attributes, we applied general data processing methods i.e. one-hot encoding and standardization, separately to categorical data and numeric data.
*For event attributes, we applied Prefix length bucketing and Aggregation encoding methods. That is to contain all the event attributes from the first n events into
the nth bucket, and aggregation encoding considers all events in each bucket, by encoding attributes into feature vectors using different aggregation functions. 

### Deep Learning-based Survival Analysis Model
DeepSurv is a well-known and influential method in the realm of Deep Learning-based Survival Analysis introduced by Jared Katzman et al [3]. DeepSurv enhances its capabilities by incorporating a deep neural network to learn intricate patterns and relationships from the data. 
DeepSurv primarily extends the Cox Proportional Hazards model, by incorporating a deep neural network to learn intricate patterns and relationships from the data. DeepSurv adopts the Maximum Likelihood Estimation (MLE) to train the model and construct the objective function with the average negative log partial likelihood with regularization. The partial likelihood is
<div align=center> 
<img width='500' src='https://github.com/Shu-Shine/Deep-Survival-Analysis-for-Remaining-Process-Runtime-Prediction/blob/main/images/equation1.jpg'/>   
</div>
where the numerator is the hazard of cases with events indexed by i and their event times at ti, and the denominator is the summed hazard of all cases j still survival at that time, which also includes the censoring data that have not dropped out yet. This is how the survival model can handle the incomplete data, by involving all the information in its objective function.  

DeepSurv is a deep feedforward neural network. The input to the network is the observed covariates data x, i.e. the feature vectors. The network propagates the data through a sequence of hidden layers with weights. The hidden layers consist of fully connected nonlinear ReLU activation functions followed by batch normalization and dropout. The final layer is a single node that performs a linear combination of the hidden features. The hyper-parameters of the network are detailed in Table 1, which were derived manually from a set of values. Adam optimizer was used for model training, without setting initial learning rate value.

### Point prediction
Another problem with applying survival analysis to the remaining runtime prediction is that the outcome of the survival model is a probability distribution, while the ultimate goal of our task is to gain the estimation of the left time in the process, which is only a single value. An additional procedure was implemented, which was to convert probability distribution into point prediction using a probability threshold, determined from grid search.

References


