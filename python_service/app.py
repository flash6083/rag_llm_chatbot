from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama
import pypdf
import docx
import os
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)
CORS(app)

# Simple embedding using TF-IDF
class SimpleEmbedder:
    def __init__(self, max_features=384):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.is_fitted = False
        self.corpus = []

    def fit(self, texts: List[str]):
        self.corpus.extend(texts)
        if len(self.corpus) > 0:
            self.vectorizer.fit(self.corpus)
            self.is_fitted = True

    def encode(self, texts: List[str]):
        if not self.is_fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts).toarray()

print("Initializing simple embedder...")
embedder = SimpleEmbedder()
print("✅ Embedder initialized")

OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'phi3:3.8b')

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': OLLAMA_MODEL,
        'embedding_model': 'TF-IDF (lightweight)'
    })

@app.route('/fit-embedder', methods=['POST'])
def fit_embedder():
    try:
        data = request.json
        texts = data.get('texts', [])
        if not texts:
            return jsonify({'success': False, 'error': 'No texts provided'}), 400
        embedder.fit(texts)
        return jsonify({'success': True, 'message': f'Embedder fitted on {len(texts)} texts'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/generate', methods=['POST'])
def generate_response():
    """Generate response using Ollama"""
    try:
        data = request.json
        query = data.get('query')
        context_docs = data.get('context_docs', [])
        model = data.get('model', OLLAMA_MODEL)
        
        if not query:
            return jsonify({'success': False, 'error': 'No query provided'}), 400
        
        # Prepare context
        context = "\n\n".join([f"Document {i+1}:\n{doc}" 
                              for i, doc in enumerate(context_docs)])
        
        # Create prompt
        prompt = f"""You are a helpful assistant for the Computer Science Department at NIT Rourkela. 
Use the following context to answer the user's question accurately and concisely.

Context:
{context}

Question: {query}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain the answer, say "I don't have information about that in my knowledge base"
- Be specific and mention faculty names, specializations, or course details when relevant
- Keep the response clear and well-structured

Answer:"""
        
        # Generate response
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                'temperature': 0.3,
                'top_k': 40,
                'top_p': 0.9,
            }
        )
        
        return jsonify({
            'success': True,
            'data': {
                'response': response['response']
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/embed', methods=['POST'])
def embed_texts():
    try:
        data = request.json
        texts = data.get('texts', [])
        if not texts:
            return jsonify({'success': False, 'error': 'No texts provided'}), 400
        embeddings = embedder.encode(texts).tolist()
        return jsonify({
            'success': True,
            'data': {'embeddings': embeddings, 'count': len(embeddings)}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ... Keep /generate and /process-document routes as before ...

if __name__ == '__main__':
    print("\n🐍 Starting Python Microservice (Simple TF-IDF Mode)...")
    print(f"🤖 Using Ollama model: {OLLAMA_MODEL}")
    print("📊 Using TF-IDF for embeddings (lightweight, no GPU needed)")
    print("🔥 Server running on http://localhost:5001\n")
    app.run(host='0.0.0.0', port=5001, debug=True)
