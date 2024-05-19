import pandas as pd

data = {
    'Name': ['Shahzad', 'Ahmad', 'Asad', 'laraib'],
    'CGPA': [3.8, 3.2, 3.5, 3.9],
    'Marks': [85, 78, 80, 88],
    'Percentage': [90, 82, 85, 92]
}

df = pd.DataFrame(data)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['CGPA', 'Marks', 'Percentage']] = scaler.fit_transform(df[['CGPA', 'Marks', 'Percentage']])

df['Success'] = df['Percentage'] > 0.85
X = df[['CGPA', 'Marks', 'Percentage']]
y = df['Success']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Save the trained model to a file
with open('student_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the model from the file
with open('student_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = [data['CGPA'], data['Marks'], data['Percentage']]
    prediction = model.predict([input_data])[0]
    return jsonify({'success': bool(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
    
   #python app.py