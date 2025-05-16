import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from http.server import BaseHTTPRequestHandler

# Load dataset once
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../model/faq.csv'))
questions = df['Question Text'].fillna("").tolist()
answers = df['Answer Text'].fillna("").tolist()
corpus = [q + " " + a for q, a in zip(questions, answers)]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)

# Load model once
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cpu")
model = model.to(device)

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get('Content-Length'))
        post_body = self.rfile.read(content_len)
        data = json.loads(post_body)
        query = data.get("query", "")

        # Similarity search
        query_vec = vectorizer.transform([query])
        similarity = cosine_similarity(query_vec, X).flatten()
        best_match_idx = similarity.argmax()
        context = corpus[best_match_idx]

        # Prompt and generate
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Respond
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"answer": response.strip()}).encode())
