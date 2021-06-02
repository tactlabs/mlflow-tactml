# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Split the data into training and test sets. (0.75, 0.25) split.
    class TactModel(object):
    
        def __init__(self, csv_path, pickle_path = None):
            
            self.csv_filepath = csv_path
            self.pickle_filepath = pickle_path
            self.model = None
            
        def make_model(self):
            
            df = pd.read_csv(self.csv_filepath)
            

            # Train and Build
            X= df.drop(df.columns[-1], axis=1)
            y = df[df.columns[-1]]
            
            dtrmodel = DecisionTreeRegressor()
            
            dtrmodel.fit(X, y)
            
            self.model = dtrmodel
        
        def save_model_to_pickle(self):
            pass
        
        def load_model(self):
            
            return self.model
        
        def predict(self, data):
            
            prediction = self.model.predict(data)
            
            return prediction[0]

    # The predicted column is "quality" which is a scalar from [3, 9]

    with mlflow.start_run():
        tact_model = TactModel(
            'abalone-final.csv'
        )

        tact_model.make_model()

        print("Tact ML created at", tact_model)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(tact_model, "model", registered_model_name="DecisionTreeRegressionAbalone")
        else:
            mlflow.sklearn.log_model(tact_model, "model")
