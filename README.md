# Deep-Survival-Analysis-for-Remaining-Process-Runtime-Prediction

## Introduction
Using neural networks for remaining time prediction of business processes has become the state-of-the-art in recent years. One direction designed for estimating the time to an event of interest has been mainly neglected in this domain so far: Survival Analysis. Survival Analysis emerged from the field of statistics, but recently also methods that combine Survival Analysis and Deep Learning have been proposed. 
This study represents the inaugural attempt to apply Deep Learning-based Survival Analysis to the domain of Remaining Process Runtime Prediction. We assess the effectiveness of this approach and analyze potential factors that could influence its performance. Through this innovative exploration, we provide a basis for future advancements in utilizing Survival Analysis techniques for remaining time prediction.

## Dataset
We conducted an experiment utilizing a real-life dataset that is publicly accessible, focusing on predicting the remaining duration of loan application processes. The historical data was sourced from a Dutch Financial Institute, which is available at the 4TU Center for Research Data[1]. The data is structured as an event log containing numerous cases, representing instances of the business process.
<div align=center>
<img width='400' src='https://github.com/Shu-Shine/Deep-Survival-Analysis-for-Remaining-Process-Runtime-Prediction/blob/main/images/Loan_applications_process.png'/>  
<div>Fig. 1 Loan applications process[2].</div>
<img width='400' src='https://github.com/Shu-Shine/Deep-Survival-Analysis-for-Remaining-Process-Runtime-Prediction/blob/main/images/event_log.png'/> 
<div>Fig. 2 Event log data.</div>
</div>

## Method
### Feature Engineering
* Firstly we introduced the event indicator, taking “Application Pending” as our event of interest.
* Then for the static case attributes, we applied general data processing methods i.e. one-hot encoding and standardization, separately to categorical data and numeric data.
* For event attributes, we applied Prefix length bucketing and Aggregation encoding methods. That is to contain all the event attributes from the first n events into
the nth bucket, and aggregation encoding considers all events in each bucket, by encoding attributes into feature vectors using different aggregation functions. 

### Deep Learning-based Survival Analysis Model
DeepSurv is a well-known and influential method in the realm of Deep Learning-based Survival Analysis introduced by Jared Katzman et al [3]. The method primarily extends the Cox Proportional Hazards model and enhances its capabilities by incorporating a deep neural network to learn intricate patterns and relationships from the data. DeepSurv adopts the Maximum Likelihood Estimation (MLE) to train the model and construct the objective function with the average negative log partial likelihood with regularization. The partial likelihood is
<div align=center> 
<img width='400' src='https://github.com/Shu-Shine/Deep-Survival-Analysis-for-Remaining-Process-Runtime-Prediction/blob/main/images/equation1.jpg'/>            
</div>
where the numerator is the hazard of cases with events indexed by i and their event times at t_i, and the denominator is the summed hazard of all cases j still survival at that time, which also includes the censoring data that have not dropped out yet. This is how the survival model can handle the incomplete data, by involving all the information in its objective function.  

DeepSurv is a deep feedforward neural network. The input to the network is the observed covariates data x, i.e. the feature vectors. The network propagates the data through a sequence of hidden layers with weights. The hidden layers consist of fully connected nonlinear ReLU activation functions followed by batch normalization and dropout. The final layer is a single node that performs a linear combination of the hidden features. The hyper-parameters of the network are detailed in Table 1, which were derived manually from a set of values. Adam optimizer was used for model training, without setting initial learning rate value.
<div align=center> 
<img width='450' src='https://github.com/Shu-Shine/Deep-Survival-Analysis-for-Remaining-Process-Runtime-Prediction/blob/main/images/Table1.jpg'/>            
</div>

### Point prediction
Another problem with applying survival analysis to the remaining runtime prediction is that the outcome of the survival model is a probability distribution, while the ultimate goal of our task is to gain the estimation of the left time in the process, which is only a single value. An additional procedure was implemented, which was to convert probability distribution into point prediction using a probability threshold, determined from grid search.

## Result
The effectiveness of DeepSurv can be described by comparing its Mean Absolute Error (MAE) with that of other models in the evaluation. In this case, DeepSurv achieved an MAE of 7.842, which overperforms the traditional methods like Transition system and Stochastic Petri net. While DeepSurv did not achieve the highest performance in this particular evaluation, being outperformed by LSTM with a score of 7.15, it still demonstrates the capability to provide reasonably accurate predictions of remaining process runtime, showcasing its potential as a viable approach in this field.  

References  
[1] Boudewijn van Dongen, Bpi challenge 2017, 2017, [online] Available: https://data.4tu.nl/articles/_/12696884/1 .  
[2] Liese Blevi, et al. Process mining on the loan application process of a Dutch Financial Institute. [online] Available: https://www.win.tue.nl/bpi/2017/bpi2017_winner_professional.pdf .  
[3] Katzman, Jared L., Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, and Yuval Kluger. ”DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network.” BMC medical research methodology 18, no. 1 (2018): 1-12.  


