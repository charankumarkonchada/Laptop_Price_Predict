import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error,mean_squared_error
from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestRegressor

MODEL_FILE="model.pkl"
PIPELINE_FILE="pipeline.pkl"

def build_pipeline(num_att,cat_att):
    num_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
    
    cat_pipeline=Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])
    
    full_pipeline=ColumnTransformer([
        ("num",num_pipeline,num_att),
        ("cat",cat_pipeline,cat_att)
    ])

    return full_pipeline
if not os.path.exists(MODEL_FILE):
    laptop=pd.read_csv("laptop_data.csv")
    
    #handling missing values
    laptop["ram_gb"]=laptop["ram_gb"].fillna(laptop["ram_gb"].mean())
    laptop["storage_gb"]=laptop["storage_gb"].fillna(laptop["storage_gb"].mean())
    laptop["storage_type"]=laptop["storage_type"].fillna(laptop["storage_type"].mode()[0])
    laptop["gpu_brand"]=laptop["gpu_brand"].fillna(laptop["gpu_brand"].mode()[0])
    laptop["weight_kg"]=laptop["weight_kg"].fillna(laptop["weight_kg"].mean())
    laptop["user_rating"]=laptop["user_rating"].fillna(laptop["user_rating"].mean())
    laptop["num_reviews"]=laptop["num_reviews"].fillna(laptop["num_reviews"].mean())
    
    bins = [0, 4, 8, 16, 64]  
    labels = ["Low", "Medium", "High", "Very High"]

    laptop["ram_category"] = pd.cut(laptop["ram_gb"],bins=bins,labels=labels)
    
    spl=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    
    for train_index,test_index in spl.split(laptop,laptop["ram_category"]):
        train_set = laptop.loc[train_index].drop("ram_category", axis=1)
        test_set = laptop.loc[test_index].drop("ram_category", axis=1)

    test_set.to_csv("test_input.csv", index=False)
    laptop = train_set
    
    #labesl and features
    lap_labels=laptop["price_inr"].copy()
    lap_features=laptop.drop("price_inr",axis=1)
    
    #num and cat columns
    num_att=lap_features.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_att=lap_features.select_dtypes(include=["object"]).columns.tolist()
    
    #normalization and encoding
    pipelines=build_pipeline(num_att,cat_att)
    l_prepared=pipelines.fit_transform(lap_features)
    
    model=LinearRegression()
    model.fit(l_prepared,lap_labels)
    
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipelines,PIPELINE_FILE)
    
    # c_laptop=laptop.copy()
    # predictions=model.predict(l_prepared)
    # c_laptop["predicted_data"]=predictions
    # c_laptop.to_csv("real_data.csv",index=False)
    print("model is trained successfully")
    
else:
    #inferences
    model=joblib.load(MODEL_FILE)
    pipelines=joblib.load(PIPELINE_FILE)
    
    test_data=pd.read_csv("test_input.csv")
    transformed_data=pipelines.transform(test_data)
    
    real_data=test_data["price_inr"]
    predictions=model.predict(transformed_data)
    # test_data["predicted_price"]=predictions
    test_data["price_inr"]=predictions
    
    test_data.to_csv("test_output.csv",index=False)
    print("rmse is",root_mean_squared_error(real_data,predictions))
    print("MSE is",mean_squared_error(real_data,predictions))
    
    print("Predicted values are stored in test_output.csv file")
    
    