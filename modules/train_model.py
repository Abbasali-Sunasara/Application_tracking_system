import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # Used to save the model to a file

# 1. GENERATE SYNTHETIC DATA (The "Mock Database")
# We create 1,000 fake resumes to teach the model what a "Good" vs "Bad" match looks like.
np.random.seed(42)
n_samples = 1000

# Feature 1: TF-IDF Match Score (Random number between 0% and 100%)
match_scores = np.random.uniform(0, 100, n_samples)

# Feature 2: JD Coverage % (Random number between 0% and 100%)
# This solves your specific concern: (Matched Skills / JD Skills)
jd_coverage = np.random.uniform(0, 100, n_samples)

# Feature 3: Word Count (Random length between 100 words and 2000 words)
word_counts = np.random.randint(100, 2000, n_samples)

# Create the DataFrame
df = pd.DataFrame({
    'match_score': match_scores,
    'jd_coverage': jd_coverage,
    'word_counts': word_counts
})

# 2. DEFINE THE "HIRE" LOGIC (The Ground Truth)
# We teach the model: "If Score is High AND Coverage is High -> HIRE"
def hire_logic(row):
    # Rule 1: If Match Score > 60% AND Coverage > 50% -> YES
    if row['match_score'] > 60 and row['jd_coverage'] > 50:
        return 1 # Hired
    
    # Rule 2: Exception for High Potential (Low Match but Great Coverage)
    # Example: A Junior who knows ALL the skills but lacks "experience" keywords
    if row['match_score'] > 40 and row['jd_coverage'] > 80:
        return 1 # Hired
        
    # Rule 3: Reject really short resumes (Lazy application)
    if row['word_counts'] < 200:
        return 0 # Reject
        
    return 0 # Reject

# Apply the logic to create the "Label" (Y)
df['hired'] = df.apply(hire_logic, axis=1)

# Add some "Real World Noise" (Because human hiring isn't perfect!)
# We flip 5% of the results to make the model robust against errors.
mask = np.random.rand(len(df)) < 0.05
df.loc[mask, 'hired'] = 1 - df.loc[mask, 'hired']

# 3. TRAIN THE MODEL
X = df[['match_score', 'jd_coverage', 'word_counts']] # The Input Features
y = df['hired'] # The Target Label

# Split into Train (80%) and Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. EVALUATE & SAVE
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"✅ Model Trained Successfully!")
print(f"📊 Model Accuracy: {accuracy * 100:.2f}%")
print("------------------------------------------------")
print("🧪 Test Case 1: Perfect Match (Score: 85, Coverage: 90)")
print(f"   Prediction: {'HIRE' if model.predict([[85, 90, 800]])[0] == 1 else 'REJECT'}")
print("------------------------------------------------")
print("🧪 Test Case 2: Your Scenario (Score: 50, Coverage: 71)")
print(f"   Prediction: {'HIRE' if model.predict([[50, 71, 600]])[0] == 1 else 'REJECT'}")

# Save the brain to a file
joblib.dump(model, 'hiring_model.pkl')
print("\n💾 Model saved as 'hiring_model.pkl'")