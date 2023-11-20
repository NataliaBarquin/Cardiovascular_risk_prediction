### DESCRIPTION OF THE DATA
----------

We analyzed a health dataset to predict the **10-year risk of coronary disease** in patients using supervised machine learning models. Our response variable was the 10-year risk of coronary disease, and our predictor variables included **sociodemographic data** (age, gender, educational level) and **medical data** (hypertension, diabetes, cholesterol level, BMI, and others).

### ANALYSIS AND TRANSFORMATION
---------------

- During the exploration of the dataset, we obtained the following information: there are 3390 records, 15 predictor variables, and one response variable.

- Some columns have null values, which we have filled using the most appropriate technique for each case.

- We have categorized numerical variables to create two new variables.

- In the exploration, we have found some interesting data. In certain variables, the samples are uniformly distributed among their distinct values, such as in gender or smoking habits. In contrast, in other variables, one value has considerably more samples than the others. For instance, the majority of records in diabetes or previous heart attack history indicate negative results.

- It appears that certain predictor variables do not affect the response variable. For instance, data indicates that the chances of developing coronary disease are the same whether you smoke or not, which is contrary to initial assumptions. Nevertheless, a diabetes diagnosis appears to increase the risk of coronary disease.

- The majority of our predictor variables are not related to each other, which is good for our purpose.

- The data distribution generally remains consistent whether the response variable is negative or positive.  However, when it comes to age, the distribution of positive responses tends to increase while negative ones remain steady. Response tends to decrease with age, indicating that the chance of developing coronary disease rises.

- We have observed how the response variable affects some of our variables more than others. For example, the blood glucose level, heart rate, and Body Mass Index (BMI) do not appear to have a significant relationship with the response variable. However, we have observed a considerable variation in the medians of age, systolic blood pressure, and the number of daily cigarettes. These variables have a positive response alignment with higher values.

- We have observed an imbalance in the number of samples for our response variable. The data set includes a considerably larger number of negative variable values than positive ones (only 15%).

### ALGORITHM RESULTS
--------------------

We have resolved the data imbalance by utilizing two techniques: Random Over Sample and TokenSMOTE. Therefore, we have emphasized the need for Random Over Sample for better results. Random Over Sample outperformed TokenSMOTE in detecting positive cases, both true and false, in all trained models. Accurately detecting positive cases is critical in our model, which deals with medical data to predict the risk of a disease.

We have used multiple prediction models and will now discuss the conclusions drawn from each model for the selected dataset.

- **LOGISTIC REGRESSION:** We have achieved acceptable metrics. Nonetheless, we will attempt new approaches since our kappa is inadequate.

- **DECISION TREE:** We have obtained satisfactory metrics by employing this method. However, the kappa value is low, so we will explore alternative approaches. 

- **RANDOM FOREST:** Using the Random Forest technique, we attained satisfactory metrics, although the kappa score could be improved.

- **XG BOOST:** We have achieved outstanding metrics using this approach. However, overfitting remains unresolved. 

The XGBoost model achieved superior metrics on the test set; however, we have been unsuccessful in correcting the overfitting that occurs. Acceptable metrics were also obtained with the Random Forest model. Consequently, we employed both models for our predictions and we compared the results obtained by each.


### CONCLUSIONS
--------------

To progress our project, we may consider the following steps:

- Experiment with new encodings for our categorical variables to attempt to achieve better metrics.

- Explore new hyperparameters in the Decision Tree, Random Forest, and XG Boost models.

- Explore new machine learning models.

- While this is a personal project, if collaborating with a medical team that provides input data, we may suggest including new variables to enhance the model. These could comprise of exercise and dietary habits, body fat percentage, waist circumference, and other relevant factors.