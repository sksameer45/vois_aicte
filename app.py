import pickle
from preprocess import clean_text

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def predict_review(review):
    review = clean_text(review)
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)
    return prediction[0]

if __name__ == "__main__":
    print("=== Fake Review Detection ===")
    user_input = input("Enter a product review: ")
    result = predict_review(user_input)
    print(f"Prediction: {result}")