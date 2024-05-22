# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:47:37 2024

@author: ikonz
"""

import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def format_datetime(dt_str):
    return datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S.%f%z')

# GPS Data
GPS_data = pd.read_csv("GPS_data.csv")
GPS_data['RECORD_TIMESTAMP'] = GPS_data['RECORD_TIMESTAMP'].apply(format_datetime)
GPS_data['RECORD_TIMESTAMP'] = GPS_data['RECORD_TIMESTAMP'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
GPS_data['RECORD_TIMESTAMP'] = pd.to_datetime(GPS_data['RECORD_TIMESTAMP'])
# Calculating the Time Difference in minutes using GPS record time of each Shippment number 
GPS_data = GPS_data.groupby('SHIPMENT_NUMBER')['RECORD_TIMESTAMP'].agg(['min', 'max'])
GPS_data['Time_Diff'] = GPS_data['max'] - GPS_data['min']
GPS_data['Time_Diff'] = GPS_data['Time_Diff'].dt.total_seconds()/60


# Shipment Data 
Shipment_bookings = pd.read_csv('Shipment_bookings.csv')

datetime_columns = [
    'FIRST_COLLECTION_SCHEDULE_EARLIEST',
    'FIRST_COLLECTION_SCHEDULE_LATEST',
    'LAST_DELIVERY_SCHEDULE_EARLIEST',
    'LAST_DELIVERY_SCHEDULE_LATEST'
]

for column in datetime_columns:
    Shipment_bookings[column] = pd.to_datetime(Shipment_bookings[column], format='%Y-%m-%dT%H:%M:%S.%f%z')
    Shipment_bookings[column] = pd.to_datetime(Shipment_bookings[column], format='%Y-%m-%d %H:%M:%S')

# Filter data between October 1st and December 31st, 2023
start_date = pd.Timestamp('2023-10-01').tz_localize('UTC')
end_date = pd.Timestamp('2023-12-31').tz_localize('UTC')

filtered_shipments = Shipment_bookings[
    (Shipment_bookings['LAST_DELIVERY_SCHEDULE_LATEST'] >= start_date) &
    (Shipment_bookings['LAST_DELIVERY_SCHEDULE_LATEST'] <= end_date)]

filtered_shipments= pd.merge(filtered_shipments, GPS_data[['Time_Diff']],on='SHIPMENT_NUMBER',how = 'inner')
filtered_shipments= filtered_shipments.rename(columns ={'Time_Diff':'Actual_timetodelivery_in_mins'})

'''
Operational teams rely heavily on KPIs like on-time collection and on-time delivery to gauge carrier performance. 
What percentage of shipments met the on-time delivery threshold (arriving no later than 30 minutes past the scheduled delivery window) between October 1st and December 31st, 2023? 
'''

filtered_shipments['timetodelivery_in_mins'] = filtered_shipments['LAST_DELIVERY_SCHEDULE_LATEST'] - filtered_shipments['FIRST_COLLECTION_SCHEDULE_EARLIEST']
filtered_shipments['timetodelivery_in_mins'] = filtered_shipments['timetodelivery_in_mins'].dt.total_seconds()/60

filtered_shipments['on_time'] = 0
filtered_shipments.loc[(filtered_shipments['Actual_timetodelivery_in_mins']-filtered_shipments['timetodelivery_in_mins']) <= 30, 'on_time'] = 1

# % of on-time deliveries
on_time_percentage = filtered_shipments['on_time'].mean() * 100
# print(f"Percentage of on-time deliveries: {on_time_percentage:.2f}%")
# Percentage of on-time deliveries: 74.03%

'''
Timely communication of potential delays is crucial for shippers. During the 3-month period from 1st Oct to 31st Dec 2023, which shipper(s) should be notified automatically regarding 
potential late delivery of which shipments, and at what times? 
'''
late_deliveries = filtered_shipments[filtered_shipments['on_time']== 0]
notifications = late_deliveries[['PROJECT_ID', 'SHIPMENT_NUMBER', 'LAST_DELIVERY_SCHEDULE_LATEST']]
notifications = notifications.rename(columns={'LAST_DELIVERY_SCHEDULE_LATEST': 'notification_time'})

# ---------------------------------------------------------------------------------------------------#

''' Predict the likelihood of delay for the list of shipments in “New_bookings.csv” dataset.   '''

# New Shipment Bookings Data 
New_bookings = pd.read_csv('New_bookings.csv')

datetime_columns = [
    'FIRST_COLLECTION_SCHEDULE_EARLIEST',
    'FIRST_COLLECTION_SCHEDULE_LATEST',
    'LAST_DELIVERY_SCHEDULE_EARLIEST',
    'LAST_DELIVERY_SCHEDULE_LATEST'
]

for column in datetime_columns:
    New_bookings[column] = pd.to_datetime(New_bookings[column], format='%Y-%m-%dT%H:%M:%S.%f%z')
    New_bookings[column] = pd.to_datetime(New_bookings[column], format='%Y-%m-%d %H:%M:%S')

# Data Preparation

New_bookings['timetodelivery_in_mins'] = New_bookings['LAST_DELIVERY_SCHEDULE_LATEST'] - New_bookings['FIRST_COLLECTION_SCHEDULE_EARLIEST']
New_bookings['timetodelivery_in_mins'] = New_bookings['timetodelivery_in_mins'].dt.total_seconds()/60

# Data Preparation
# VECHICLE BUILD UP less than 100 count in the data , re-categorized as others in the training data (Shipment data)
value_counts = filtered_shipments['VEHICLE_BUILD_UP'].value_counts()
vc = filtered_shipments['VEHICLE_BUILD_UP'].isin(value_counts[value_counts <= 100].index)
filtered_shipments.loc[vc, 'VEHICLE_BUILD_UP'] = 'others'

value_counts = filtered_shipments['VEHICLE_SIZE'].value_counts()
vc = filtered_shipments['VEHICLE_SIZE'].isin(value_counts[value_counts <= 100].index)
filtered_shipments.loc[vc, 'VEHICLE_SIZE'] = 'others'

# filtered_shipments['on_time'] = filtered_shipments['on_time'].astype(object)

# Feature engineering
# Day of week 
filtered_shipments['DAY_OF_WEEK'] = filtered_shipments['LAST_DELIVERY_SCHEDULE_LATEST'].dt.dayofweek
filtered_shipments['DAY_OF_WEEK'] = filtered_shipments['DAY_OF_WEEK'].astype(object)

# Function to categorize the time of day
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    elif 21 <= hour < 24 or 0 <= hour < 5:
        return 'Late Evening'
    else:
        return 'Unknown'
filtered_shipments['TIME_OF_DAY'] = filtered_shipments['LAST_DELIVERY_SCHEDULE_LATEST'].dt.hour.apply(get_time_of_day)

columns_to_encode = ['VEHICLE_BUILD_UP', 'VEHICLE_SIZE','TIME_OF_DAY']
encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' is optional to avoid dummy variable trap
encoded_data = encoder.fit_transform(filtered_shipments[columns_to_encode])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns_to_encode))
filtered_shipments = pd.concat([filtered_shipments, encoded_df], axis=1)

columns_to_drop = ['PROJECT_ID', 'SHIPMENT_NUMBER', 'CARRIER_DISPLAY_ID','FIRST_COLLECTION_POST_CODE',
                    'LAST_DELIVERY_POST_CODE', 'FIRST_COLLECTION_LATITUDE','VEHICLE_BUILD_UP','VEHICLE_SIZE',
                    'FIRST_COLLECTION_LONGITUDE', 'LAST_DELIVERY_LATITUDE',
                    'LAST_DELIVERY_LONGITUDE', 'FIRST_COLLECTION_SCHEDULE_EARLIEST',
                    'FIRST_COLLECTION_SCHEDULE_LATEST', 'LAST_DELIVERY_SCHEDULE_EARLIEST',
                    'LAST_DELIVERY_SCHEDULE_LATEST','Actual_timetodelivery_in_mins','TIME_OF_DAY','VEHICLE_BUILD_UP_Box 44ft','VEHICLE_BUILD_UP_LWB Van']
train_dataset = filtered_shipments.drop(columns=columns_to_drop)

X = train_dataset.drop('on_time', axis=1)
y = train_dataset['on_time']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# Train and evaluate each classifier
results = []
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append((name, accuracy))
    print(f'Classifier: {name}')
    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(y_test, y_pred))
    print('-' * 30)

# Display results
results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy'])
print(results_df)

lr = LogisticRegression()
lr = lr.fit(X_train, y_train)

# VECHICLE BUILD UP less than 100 count in the data , re-categorized as others in the new bookings data
value_counts = New_bookings['VEHICLE_BUILD_UP'].value_counts()
vc = New_bookings['VEHICLE_BUILD_UP'].isin(value_counts[value_counts <= 100].index)
New_bookings.loc[vc, 'VEHICLE_BUILD_UP'] = 'others'

value_counts = New_bookings['VEHICLE_SIZE'].value_counts()
vc = New_bookings['VEHICLE_SIZE'].isin(value_counts[value_counts <= 100].index)
New_bookings.loc[vc, 'VEHICLE_SIZE'] = 'others'

New_bookings['DAY_OF_WEEK'] = New_bookings['LAST_DELIVERY_SCHEDULE_LATEST'].dt.dayofweek
New_bookings['DAY_OF_WEEK'] = New_bookings['DAY_OF_WEEK'].astype(object)
New_bookings['TIME_OF_DAY'] = New_bookings['LAST_DELIVERY_SCHEDULE_LATEST'].dt.hour.apply(get_time_of_day)

encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' is optional to avoid dummy variable trap
encoded_data = encoder.fit_transform(New_bookings[columns_to_encode])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns_to_encode))
New_bookings = pd.concat([New_bookings, encoded_df], axis=1)
New_bookings_val = New_bookings.drop(columns=columns_to_drop, errors='ignore')
col= ['CARRIER_ID','SHIPPER_ID','SHIPMENT_NUMBER', 'VEHICLE_SIZE', 'VEHICLE_BUILD_UP',
       'FIRST_COLLECTION_POST_CODE', 'LAST_DELIVERY_POST_CODE',
       'FIRST_COLLECTION_LATITUDE', 'FIRST_COLLECTION_LONGITUDE',
       'LAST_DELIVERY_LATITUDE', 'LAST_DELIVERY_LONGITUDE',
       'FIRST_COLLECTION_SCHEDULE_EARLIEST',
       'FIRST_COLLECTION_SCHEDULE_LATEST', 'LAST_DELIVERY_SCHEDULE_EARLIEST',
       'LAST_DELIVERY_SCHEDULE_LATEST','TIME_OF_DAY']
New_bookings_val = New_bookings.drop(columns=col, errors='ignore')
y_pred_newbooking = lr.predict(New_bookings_val)

# Add the predictions to the original test data
New_bookings['predictions'] = y_pred_newbooking



