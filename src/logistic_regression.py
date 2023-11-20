import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score , cohen_kappa_score, roc_curve,roc_auc_score



class Logistic_regression():

    """Apply a Logistic Regression model and calculate the importance of predictor variables for the model.
    """

    def __init__(self, dataframe, response_variable, title_dataframe):

        """Constructor method.
        Parameters: Dataframe for application of model, response variable and title.
        """

        self.dataframe = dataframe
        self.response_variable = response_variable
        self.title_dataframe = title_dataframe


    def fit_model(self):

        """Fits a logistic regression model on the given dataset, dividing it into separate training and testing sets.
        Parameters: None
        Returns: None
        """

        X = self.dataframe.drop(self.response_variable, axis = 1)
        y = self.dataframe[self.response_variable]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

        self.log_reg = LogisticRegression(n_jobs=-1, max_iter = 1000)

        self.log_reg.fit(self.x_train, self.y_train)

        self.y_pred_train = self.log_reg.predict(self.x_train)
        self.y_pred_test = self.log_reg.predict(self.x_test)


    def print_confusion_matrix(self):

        """Print confusion matrix.
        Parameters: None
        Returns: None
        """

        mat_lr1 = confusion_matrix(self.y_test, self.y_pred_test)

        my_heatmap_palette = LinearSegmentedColormap.from_list('Gradient', ['lightcyan', 'mediumturquoise', 'teal'], N=1000)

        plt.figure(figsize = (6, 6))
        sns.heatmap(mat_lr1, square=True, annot=True, fmt="d", cmap = my_heatmap_palette)

        plt.title(f'{self.title_dataframe.upper()} CONFUSION MATRIX', fontsize = 12, color = 'darkslategray', fontweight = "bold")
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        plt.show()


    def get_metrics(self, model_name):

        """Create a dataframe showing the model's metrics, divided between the train and test sets.
        Parameters: None
        Returns: Dataframe containing model metrics.
        """
        
        accuracy_test = accuracy_score(self.y_test, self.y_pred_test)
        precision_test = precision_score(self.y_test, self.y_pred_test)
        recall_test = recall_score(self.y_test, self.y_pred_test)
        f1_test = f1_score(self.y_test, self.y_pred_test)
        kappa_test = cohen_kappa_score(self.y_test, self.y_pred_test)

        accuracy_train = accuracy_score(self.y_train, self.y_pred_train)
        precision_train = precision_score(self.y_train, self.y_pred_train)
        recall_train = recall_score(self.y_train, self.y_pred_train)
        f1_train = f1_score(self.y_train, self.y_pred_train)
        kappa_train = cohen_kappa_score(self.y_train, self.y_pred_train)
            
        dataframe = pd.DataFrame({'accuracy': [accuracy_test, accuracy_train], 
                            'precision': [precision_test, precision_train],
                            'recall': [recall_test, recall_train], 
                            'f1': [f1_test, f1_train],
                            'kappa': [kappa_test, kappa_train],
                            'set': ['test', 'train']})
        
        dataframe['model'] = model_name

        return dataframe   
    

    def create_df_feature_importance(self):

        """Create a dataframe showing the importance of the features for the model.
        Parameters: None
        Returns: Dataframe containing model the importance of the features.
        """

        coefficients = self.log_reg.coef_[0]

        self.predictors_sig_dataframe = pd.DataFrame({'predictor': self.x_train.columns, 'coeficiente': coefficients})

        self.predictors_sig_dataframe['importance'] = abs(self.predictors_sig_dataframe['coeficiente'])
        self.predictors_sig_dataframe.sort_values(by='importance', ascending=False, inplace=True)

        return self.predictors_sig_dataframe


    def create_barplot_feature_importance(self):

        """Create a barplot showing the importance of the features for the model.
        Parameters: None
        Returns: None
        """

        my_barplot_palette = LinearSegmentedColormap.from_list('Gradient', ['teal', 'mediumturquoise', 'paleturquoise'])
        num_bars = (len(self.dataframe.columns) -1)

        plt.figure(figsize=(10,6))
        sns.barplot(x = 'importance', y = 'predictor', data = self.predictors_sig_dataframe, palette= my_barplot_palette(np.linspace(0, 1, num_bars)))
        plt.title(f'{self.title_dataframe.upper()} LOGISTIC REGRESSION FEATURE IMPORTANCE', fontsize = 12, color = 'darkslategray', fontweight = 'bold')
        plt.show()