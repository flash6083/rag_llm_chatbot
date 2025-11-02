from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama
import os
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import re

app = Flask(__name__)
CORS(app)

# Configuration
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'phi3:3.8b')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # 384 dimensions, fast, accurate, free

# Initialize embedding model
print("🔄 Initializing embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print(f"✅ Loaded: {EMBEDDING_MODEL} ({embedder.get_sentence_embedding_dimension()} dimensions)")


def preprocess_text(text: str) -> str:
    """
    Clean and normalize text for better embedding quality
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
    return text.strip()


def expand_query(query: str) -> str:
    """
    Expand query with synonyms for better retrieval
    """
    query_lower = query.lower()
    
    # Academic domain expansions
    expansions = {
        'faculty': ['professor', 'teacher', 'instructor', 'staff', 'lecturer'],
        'course': ['subject', 'class', 'curriculum', 'paper'],
        'research': ['publication', 'project', 'work', 'area', 'interest'],
        'phd': ['doctorate', 'doctoral', 'research scholar', 'ph.d'],
        'mtech': ['m.tech', 'masters', 'postgraduate'],
        'btech': ['b.tech', 'bachelor', 'undergraduate'],
        'contact': ['email', 'phone', 'office', 'reach'],
        'teach': ['teaching', 'teaches', 'instructor', 'course'],
        'lab': ['laboratory', 'research group', 'facility'],
    }
    
    expanded_terms = [query]
    for key, synonyms in expansions.items():
        if key in query_lower:
            expanded_terms.extend(synonyms)
    
    return ' '.join(expanded_terms)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ollama_model': OLLAMA_MODEL,
        'embedding_model': EMBEDDING_MODEL,
        'embedding_dimensions': embedder.get_sentence_embedding_dimension(),
        'service': 'Python Microservice (Embeddings + LLM)'
    })


@app.route('/embed', methods=['POST'])
def embed_texts():
    """
    Generate embeddings for texts using sentence-transformers
    Better than TF-IDF because it captures semantic meaning
    """
    try:
        data = request.json
        texts = data.get('texts', [])
        normalize = data.get('normalize', True)  # Normalize for cosine similarity
        
        if not texts:
            return jsonify({'success': False, 'error': 'No texts provided'}), 400
        
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Generate embeddings
        embeddings = embedder.encode(
            processed_texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            batch_size=32
        )
        
        return jsonify({
            'success': True,
            'data': {
                'embeddings': embeddings.tolist(),
                'count': len(embeddings),
                'dimensions': embeddings.shape[1]
            }
        })
    
    except Exception as e:
        print(f"❌ Embedding error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/embed-query', methods=['POST'])
def embed_query():
    """
    Embed a search query with expansion for better retrieval
    """
    try:
        data = request.json
        query = data.get('query', '')
        use_expansion = data.get('expand', True)
        
        if not query:
            return jsonify({'success': False, 'error': 'No query provided'}), 400
        
        # Optionally expand query
        if use_expansion:
            expanded = expand_query(query)
            print(f"📝 Query expansion: '{query}' → '{expanded}'")
        else:
            expanded = query
        
        # Clean and embed
        processed = preprocess_text(expanded)
        embedding = embedder.encode(
            [processed],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]
        
        return jsonify({
            'success': True,
            'data': {
                'embedding': embedding.tolist(),
                'original_query': query,
                'expanded_query': expanded if use_expansion else None,
                'dimensions': len(embedding)
            }
        })
    
    except Exception as e:
        print(f"❌ Query embedding error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/rerank', methods=['POST'])
def rerank_results():
    """
    Re-rank search results using hybrid scoring
    Combines vector similarity + keyword matching + other signals
    """
    try:
        data = request.json
        query = data.get('query', '')
        results = data.get('results', [])  # [{text, score, metadata}]
        top_k = data.get('top_k', 5)
        
        if not query or not results:
            return jsonify({'success': False, 'error': 'Query and results required'}), 400
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score each result
        for result in results:
            text_lower = result['text'].lower()
            text_words = len(result['text'].split())
            
            # 1. Vector similarity (from ChromaDB)
            vector_score = 1 - result.get('score', 0)  # Convert distance to similarity
            
            # 2. Keyword overlap
            text_word_set = set(text_lower.split())
            keyword_overlap = len(query_words & text_word_set) / len(query_words) if query_words else 0
            
            # 3. Exact phrase match bonus
            exact_match_bonus = 0.15 if query_lower in text_lower else 0
            
            # 4. Length penalty (prefer concise, relevant chunks)
            length_penalty = min(1.0, 250 / max(text_words, 1))
            
            # 5. Position bonus (if metadata has chunk_id)
            position_bonus = 0
            if result.get('metadata', {}).get('chunk_id') == 0:
                position_bonus = 0.05  # First chunk often has key info
            
            # Combined score with weights
            final_score = (
                vector_score * 0.50 +          # Semantic similarity (most important)
                keyword_overlap * 0.25 +       # Keyword matching
                exact_match_bonus +            # Exact phrase bonus
                length_penalty * 0.05 +        # Prefer concise answers
                position_bonus                 # First chunk bonus
            )
            
            result['final_score'] = final_score
            result['score_breakdown'] = {
                'vector': round(vector_score, 3),
                'keyword': round(keyword_overlap, 3),
                'exact_match': exact_match_bonus,
                'length': round(length_penalty, 3),
                'position': position_bonus
            }
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'data': {
                'results': results[:top_k],
                'total_scored': len(results)
            }
        })
    
    except Exception as e:
        print(f"❌ Reranking error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate_response():
    """
    Generate response using Ollama with improved prompting
    """
    try:
        data = request.json
        query = data.get('query')
        context_docs = data.get('context_docs', [])  # List of relevant docs
        model = data.get('model', OLLAMA_MODEL)
        include_sources = data.get('include_sources', False)
        
        if not query:
            return jsonify({'success': False, 'error': 'No query provided'}), 400
        
        # Build context with relevance indicators
        if context_docs:
            context_parts = []
            for i, doc in enumerate(context_docs, 1):
                # Extract metadata if available
                metadata = doc.get('metadata', {})
                score = doc.get('final_score', doc.get('score', 0))
                
                # Format context entry
                relevance_indicator = f"[Relevance: {score:.2f}]" if score > 0 else ""
                context_parts.append(f"Context {i} {relevance_indicator}:\n{doc['text']}")
            
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "No specific context provided."
        
        # Enhanced prompt with strict instructions
        prompt = f"""You are an intelligent assistant for the Computer Science Department at NIT Rourkela.

CONTEXT INFORMATION (ordered by relevance):
{context}

USER QUESTION: {query}

CRITICAL INSTRUCTIONS:
1. Answer ONLY based on the context provided above
2. If the context doesn't contain enough information, respond: "I don't have sufficient information in my knowledge base to answer that question accurately."
3. Be specific and precise - mention exact names, numbers, specializations, course codes when available
4. Structure your answer clearly with proper paragraphs
5. If multiple relevant facts exist, synthesize them into a coherent response
6. DO NOT make assumptions or add information not present in the context
7. If asked about specific people, courses, or details, cite them accurately from context

RESPONSE FORMAT:
- Start directly with the answer (no "Based on the context..." preambles)
- Use natural, conversational language
- Be concise but complete
- End with relevant additional details if available

YOUR ANSWER:"""
        
        # Generate response with optimized parameters
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                'temperature': 0.1,      # Very low for factual accuracy
                'top_k': 20,             # Reduced for more focused responses
                'top_p': 0.85,           # Nucleus sampling
                'num_predict': 512,      # Max tokens
                'stop': ['\n\nUSER:', '\n\nQuestion:', 'Context'],  # Stop sequences
            }
        )
        
        answer = response['response'].strip()
        
        # Optionally include source information
        result = {
            'response': answer,
            'model_used': model,
            'context_chunks': len(context_docs)
        }
        
        if include_sources and context_docs:
            result['sources'] = [
                {
                    'text': doc['text'][:200] + '...' if len(doc['text']) > 200 else doc['text'],
                    'score': doc.get('final_score', doc.get('score', 0)),
                    'metadata': doc.get('metadata', {})
                }
                for doc in context_docs[:3]  # Top 3 sources
            ]
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        print(f"❌ Generation error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/analyze-query', methods=['POST'])
def analyze_query():
    """
    Analyze query complexity and suggest optimal retrieval parameters
    """
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'success': False, 'error': 'No query provided'}), 400
        
        query_lower = query.lower()
        
        # Determine query type
        query_type = 'general'
        if any(word in query_lower for word in ['who', 'name', 'faculty', 'professor']):
            query_type = 'person_search'
        elif any(word in query_lower for word in ['what', 'which', 'course', 'subject']):
            query_type = 'info_search'
        elif any(word in query_lower for word in ['how', 'explain', 'describe']):
            query_type = 'explanation'
        elif any(word in query_lower for word in ['list', 'all', 'every']):
            query_type = 'list_query'
        
        # Suggest parameters
        suggestions = {
            'person_search': {'top_k': 3, 'expand_query': True},
            'info_search': {'top_k': 5, 'expand_query': True},
            'explanation': {'top_k': 5, 'expand_query': False},
            'list_query': {'top_k': 10, 'expand_query': True},
            'general': {'top_k': 5, 'expand_query': True}
        }
        
        return jsonify({
            'success': True,
            'data': {
                'query_type': query_type,
                'suggested_params': suggestions[query_type],
                'query_length': len(query.split()),
                'complexity': 'high' if len(query.split()) > 10 else 'medium' if len(query.split()) > 5 else 'low'
            }
        })
    
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Python Microservice: Embeddings + LLM Generation")
    print("="*70)
    print(f"🤖 Ollama Model: {OLLAMA_MODEL}")
    print(f"🧠 Embedding Model: {EMBEDDING_MODEL}")
    print(f"📊 Embedding Dimensions: {embedder.get_sentence_embedding_dimension()}")
    print(f"💾 Vector Storage: Handled by Node.js (ChromaDB)")
    print("="*70)
    print("📡 Endpoints:")
    print("   POST /embed              - Generate embeddings for texts")
    print("   POST /embed-query        - Embed query with expansion")
    print("   POST /rerank             - Re-rank search results")
    print("   POST /generate           - Generate LLM response")
    print("   POST /analyze-query      - Analyze query for optimal params")
    print("   GET  /health             - Health check")
    print("="*70)
    print("🌐 Server: http://localhost:5001")
    print("="*70 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=True)