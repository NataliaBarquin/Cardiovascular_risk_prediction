import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score , cohen_kappa_score, roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV



class Decision_tree():

    """Apply a Decission Tree model and calculate the importance of predictor variables for the model.
    """

    def __init__(self, dataframe, response_variable):

        """Constructor method.
        Parameters: Dataframe for application of model, response variable.
        """

        self.dataframe = dataframe
        self.response_variable = response_variable


    def _print_tree(self, the_tree):

        """Print tree. This method is not been called out of the class.
        Parameters: None
        Returns: None
        """

        plt.figure(figsize = (40, 20))
        tree.plot_tree(the_tree, feature_names = self.feat_names, filled = True)
        plt.show()
        
        
    def fit_model(self):

        """Fits a decission tree model on the given dataset, dividing it into separate training and testing sets.
        Parameters: None
        Returns: Dictionary with max_features and max_depth.
        """

        X = self.dataframe.drop(self.response_variable, axis = 1)
        y = self.dataframe[self.response_variable]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

        my_tree = DecisionTreeClassifier(random_state =0)

        my_tree.fit(self.x_train, self.y_train)
        
        self.feat_names = list(self.x_train.columns)

        self._print_tree(my_tree)

        self.y_pred_test = my_tree.predict(self.x_test)
        self.y_pred_train = my_tree.predict(self.x_train)

        dicc_params = {'max_features': np.sqrt(len(self.x_train.columns)), 'max_depth': my_tree.tree_.max_depth}

        return dicc_params
    

    def get_metrics(self, model_name):

        """Create a dataframe showing the model's metrics, divided between the train and test sets.
        Parameters: Name of the method we are going to use.
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
    

    def define_best_model(self, parameters):

        """Fits the best model to apply.
        Parameters: The parameters that we choose for the method.
        Returns: the best model object.
        """

        grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state= 42), param_grid= parameters, cv=10, verbose= False) 

        grid_search.fit(self.x_train, self.y_train)

        self.best_tree = grid_search.best_estimator_

        self._print_tree(self.best_tree)

        return self.best_tree


    def apply_best_model(self, model_name):

        """Apply the best model defined in the previous method, and create a dataframe showing the model's metrics with the best tree.
        Parameters: Title of the dataframe we are using.
        Returns: Dataframe containing model metrics.
        """

        self.y_pred_test = self.best_tree.predict(self.x_test)
        self.y_pred_train = self.best_tree.predict(self.x_train)

        best_results_dataframe = self.get_metrics(model_name)

        return best_results_dataframe 


    def create_df_feature_importance(self):

        """Create a dataframe showing the importance of the features for the model.
        Parameters: None
        Returns: Dataframe containing model the importance of the features.
        """

        self.predictors_sig_dataframe = pd.DataFrame({'predictor': self.x_train.columns, 'importance': self.best_tree.feature_importances_})

        self.predictors_sig_dataframe.sort_values(by=['importance'], ascending=False, inplace = True)

        return self.predictors_sig_dataframe
        

    def create_barplot_feature_importance(self, title_dataframe):

        """Create a barplot showing the importance of the features for the model.
        Parameters: Title of the dataframe we are using.
        Returns: None
        """

        my_barplot_palette = LinearSegmentedColormap.from_list('Gradient', ['teal', 'mediumturquoise', 'paleturquoise'])
        num_bars = (len(self.dataframe.columns) -1)

        plt.figure(figsize=(10,6))
        sns.barplot(x = 'importance', y = 'predictor', data = self.predictors_sig_dataframe, palette= my_barplot_palette(np.linspace(0, 1, num_bars)))
        plt.title(f'{title_dataframe.upper()} DECISION TREE FEATURE IMPORTANCE', fontsize = 12, color = 'darkslategray', fontweight = 'bold')
        plt.show()
