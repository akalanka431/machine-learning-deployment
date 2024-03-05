import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import json
import httpx,os,datetime
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import numpy as np


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def train_the_model(data):
    try:     
        new_data = data
        encoders = load('encoders.joblib')
        xgb_model = load('xgb_model.joblib')
        selected_columns = ['customer_name', 'customer_address', 'customer_phone',
                            'customer_email', 'cod', 'weight', 'origin_city.name',
                            'destination_city.name', 'status.name']
        new_data_filled = new_data[selected_columns].fillna('Missing')
        for col, encoder in encoders.items():
            if col in new_data_filled.columns:
                unseen_categories = set(new_data_filled[col]) - set(encoder.classes_)
                if unseen_categories:
                    for category in unseen_categories:
                        encoder.classes_ = np.append(encoder.classes_, category)
                    new_data_filled[col] = encoder.transform(new_data_filled[col])
                else:
                    new_data_filled[col] = encoder.transform(new_data_filled[col])
        X_new = new_data_filled.drop('status.name', axis=1)
        y_new = new_data_filled['status.name']

        X_train, X_test, y_train, y_test = train_test_split(X_new,y_new, test_size=0.2, random_state=42)
        
        xgb_model.fit(X_new, y_new)
        dump(xgb_model,'xgb_model.joblib')
        
        print("Model updated with new data.")
        y_pred = xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
    except:
        data = data
        
        # Select columns
        selected_columns = ['customer_name', 'customer_address', 'customer_phone',
                            'customer_email', 'cod', 'weight',
                            'origin_city.name', 'destination_city.name', 'status.name']
        
        # Handling missing values
        data_filled = data[selected_columns].fillna('Missing')
        
        # Encoding categorical variables
        encoders = {col: LabelEncoder() for col in selected_columns if data_filled[col].dtype == 'object'}
        for col, encoder in encoders.items():
            data_filled[col] = encoder.fit_transform(data_filled[col])
        
        # Splitting the dataset
        X = data_filled.drop('status.name', axis=1)
        y = data_filled['status.name']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Setup the hyperparameter grid to search
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.4],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1],
            'colsample_bytree': [0.3, 0.7]
        }
        
        # Initialize the classifier
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        
        # Setup GridSearchCV
        grid_search = GridSearchCV(xgb, param_grid, cv=2, n_jobs=-1, scoring='accuracy')
        
        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)
        
        # Get the best parameters
        best_params = grid_search.best_params_
        print("Best parameters:", best_params)
        
        # Train the model with best parameters
        best_xgb = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
        best_xgb.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = best_xgb.predict(X_test)
        y_pred_proba = best_xgb.predict_proba(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        
        # Print the results
        print("Accuracy:", accuracy)
        print("Classification Report:\n", classification_report(y_test, y_pred))
    
        
        # Save the model
        model_filename = 'xgb_model.joblib'
        dump(best_xgb, model_filename)
        
        # Save the encoders
        encoders_filename = 'encoders.joblib'
        dump(encoders, encoders_filename)
        
        print(f"Model saved as {model_filename}")
        print(f"Encoders saved as {encoders_filename}")
        print("new base model trained")
    
@app.get("/trigger_the_data_fecher")
async def your_continuous_function(page: int,paginate: int,Tenant: str):
    print("data fetcher running.....")
            
    # Initialize an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()
            
    # Update the payload for each page
    url = "https://dev3.api.curfox.parallaxtec.com/api/ml/order-list?sort=id&paginate="+str(paginate)+"&page="+str(page)
            
    payload = {}
    headers = {
                    'Accept': 'application/json',
                    'X-Tenant': Tenant #'royalexpress'
                  }
            
    response = requests.request("GET", url, headers=headers, data=payload)
            
    # Sample JSON response
    json_response = response.json()
    # Extracting 'data' for conversion
    data = json_response['data']
    data_count = len(data)  
    
    df = pd.json_normalize(data)
    
            
    # Concatenate the current page's DataFrame with the combined DataFrame
    combined_df = pd.concat([combined_df, df], ignore_index=True)
            
    data = combined_df[combined_df['status.name'].isin(['RETURN TO CLIENT', 'DELIVERED'])]
    print("data collected from page : "+str(page))
    #data.to_csv("new.csv")
    
    train_the_model(data)

    return "model trained with page number: "+str(page)+" data count :"+str(data_count)

@app.get("/get_latest_model_updated_time")
async def model_updated_time():
    m_time_encoder = os.path.getmtime('encoders.joblib')
    m_time_model = os.path.getmtime('xgb_model.joblib')
    return {"base model created time ":datetime.datetime.fromtimestamp(m_time_encoder),
            "last model updated time":datetime.datetime.fromtimestamp(m_time_model)}
