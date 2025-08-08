import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv('dreaddit-train.csv', encoding='ISO-8859-1')

# Drop unnecessary columns
df.drop(['text', 'post_id', 'sentence_range', 'id', 'social_timestamp'], axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# **Inspect sentiment column**
print(df['sentiment'].describe())  # Check min, max, mean values

# **OPTIONAL: Discretize sentiment column**
# Let's say sentiment ranges from -1 to +1. We'll classify as:
# sentiment > 0 => 1 (Positive), sentiment <= 0 => 0 (Negative)

df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x > 0 else 0)

# Features and target
X = df.drop('sentiment', axis=1)
y = df['sentiment']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open('stresslevel.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully as stresslevel.pkl")
