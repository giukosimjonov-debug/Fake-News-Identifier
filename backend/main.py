from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hamma joydan kirishga ruxsat
    allow_methods=["*"],
    allow_headers=["*"],
)

class CheckNewsNN(nn.Module):
    def __init__(self, input_size=5000, hidden_size1=512, hidden_size2=256, num_classes=2):
        super(CheckNewsNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

model = CheckNewsNN()
model.load_state_dict(torch.load("models/fake_news_model.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    
    word_tokens = word_tokenize(text)
    filtered_text = [lemmatizer.lemmatize(w) for w in word_tokens if w not in stop_words]
    return " ".join(filtered_text)

class NewsRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_news(request: NewsRequest):

    cleaned_text = preprocess_text(request.text)
    
    vec = vectorizer.transform([cleaned_text])
    tensor_input = torch.tensor(vec.toarray(), dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(tensor_input)
        probs = F.softmax(outputs, dim=1)
        fake_prob = probs[0][0].item() * 100
        real_prob = probs[0][1].item() * 100
        
    prediction = "Real" if real_prob > fake_prob else "Fake"
    
    return {
        "prediction": prediction,
        "real_probability": round(real_prob, 2),
        "fake_probability": round(fake_prob, 2)
    }