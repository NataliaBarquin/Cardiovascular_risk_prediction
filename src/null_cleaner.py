import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer



class Null_cleaner:

    """Fill in missing values within the dataframe.
    """

    def __init__(self, dataframe):

        """Constructor method.
        Parameters: Dataframe for application of model, response variable.
        """

        self.dataframe = dataframe


    def _clean_num_null_values_with_knn(self):

        """Apply the KNN imputation method to numeric columns in the dataframe.
        Args:
            dataframe: dataframe where we want to apply the method.
        Returns:
            dataframe: dataframe whitout numeric missing values.
        """

        numerical_dataframe = self.dataframe.select_dtypes(include = np.number)

        imputerKNN = KNNImputer(n_neighbors=5)
        imputerKNN.fit(numerical_dataframe)
        numerical_knn = imputerKNN.transform(numerical_dataframe)
        knn_imputer_dataframe = pd.DataFrame(numerical_knn, columns = numerical_dataframe.columns)

        knn_columns = knn_imputer_dataframe.columns
        self.dataframe.drop(knn_columns, axis = 1, inplace = True)
        self.dataframe[knn_columns] = numerical_knn

        return self.dataframe


    def _clean_cat_null_values_with_mode(self):
        
        """For categorical columns in the dataframe, if the percentage of null values is more than 10%,
        replace them with a new category called 'Unknown'. Otherwise, replace them with the mode.
        Args:
            dataframe: dataframe where we want to apply the method.
        Returns:
            dataframe: dataframe whitout categorical missing values.
        """
                
        categorical_dataframe = self.dataframe.select_dtypes(include = ['O', 'category'])

        cat_cols_with_nulls = categorical_dataframe.columns[categorical_dataframe.isnull().any()].tolist()

        for col in cat_cols_with_nulls:
            
            null_percentage = self.dataframe[col].isna().mean() * 100 

            if null_percentage  >= 10:
                self.dataframe[col].fillna('Unknown', inplace = True)
            else:
                self.dataframe[col].fillna(self.dataframe[col].mode()[0], inplace = True)

        return self.dataframe


    def find_and_clean_all_null_values(self):

        """Use the previous functions to fill any null values in the dataframe with a single call.
        Args:
            dataframe: dataframe for null value imputation.
        Returns:
            dataframe: dataframe whitout missing values.
        """
        
        dataframe_result = self._clean_num_null_values_with_knn()
        dataframe_result = self._clean_cat_null_values_with_mode()

        return dataframe_result