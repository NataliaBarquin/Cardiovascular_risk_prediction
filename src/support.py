import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

class Cleaning_nulls:

    def __init__(self, dataframe):

        self.dataframe = dataframe


    def _find_columns_with_nulls(self):
    
        """Identify the columns in the dataset that contain missing values and make a list of these columns.
        Args:
            dataframe: dataframe where we want to find missing values.
        Returns:
            list: list of columns with missing values.
        """
            
        null_cols = []    

        null_df = self.dataframe.stb.missing()
        null_df = null_df[null_df['percent'] > 0].reset_index()

        for nombre_col in null_df['index']:
            null_cols.append(nombre_col)

        return null_cols


    def _knn_imputer(self):
    
        """Apply the KNN IMPUTER method.
        Args:
            dataframe: dataframe where we want to apply the method.
        Returns:
            dataframe: dataframe whitout missing values.
        """

        imputerKNN = KNNImputer(n_neighbors=5)
        imputerKNN.fit(self.dataframe)
        dataframe_knn= imputerKNN.transform(self.dataframe)

        df_knn_imputer = pd.DataFrame(dataframe_knn, columns = self.dataframe.columns)
        columnas_knn = df_knn_imputer.columns
        self.dataframe.drop(columnas_knn, axis = 1, inplace = True)
        self.dataframe[columnas_knn] = dataframe_knn
        
        return self.dataframe


    def find_and_clean_null_values(self):

        """Apply the KNN IMPUTER method and round the result for non-decimal values.
        Args:
            dataframe: dataframe for null value imputation.
        Returns:
            dataframe: dataframe whitout missing values.
        """
        
        columns_with_nulls = self.find_columns_with_nulls(self.dataframe)
        dataframe_result = self.knn_imputer(self.dataframe)
        dataframe_result[columns_with_nulls] = self.dataframe[columns_with_nulls].round()

        return dataframe_result




