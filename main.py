import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

data = pd.read_csv('ab_test_data.csv')
print(data.head())
print(data.isnull().sum())
data.dropna(inplace=True)

X = data.drop(columns=['userid'])
y = data['retention_7']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['sum_gamerounds']
categorical_features = ['version', 'retention_1']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


X_test_with_version = data.drop(columns=['userid', 'retention_7'])
y_test_actual = data['retention_7']


X_test_preprocessed = preprocessor.transform(X_test_with_version)
y_test_pred = model.predict(X_test_preprocessed)

X_test_with_version['predicted_retention'] = y_test_pred

retention_rate_version = X_test_with_version.groupby('version')['predicted_retention'].mean()

print(retention_rate_version)

better_version = retention_rate_version.idxmax()
print(f"The better version is: {better_version}")

joblib.dump(model, 'ab_test_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
