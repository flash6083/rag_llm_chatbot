import dotenv from 'dotenv';
import path from 'path';

dotenv.config();

export const config = {
  env: process.env.NODE_ENV || 'development',
  port: parseInt(process.env.PORT || '8000', 10),
  pythonServiceUrl: process.env.PYTHON_SERVICE_URL || 'http://localhost:5001',
  
  paths: {
    uploadDir: path.resolve(process.env.UPLOAD_DIR || './uploads'),
    knowledgeBaseDir: path.resolve(process.env.KNOWLEDGE_BASE_DIR || './knowledge_base'),
    chromaDbPath: path.resolve(process.env.CHROMA_DB_PATH || './chroma_db'),
  },
  
  ollama: {
    model: process.env.OLLAMA_MODEL || 'phi3:3.8b',
    baseUrl: process.env.OLLAMA_BASE_URL || 'http://localhost:11434',
  },
  
  vectorStore: {
    chunkSize: parseInt(process.env.CHUNK_SIZE || '500', 10),
    chunkOverlap: parseInt(process.env.CHUNK_OVERLAP || '50', 10),
    topKResults: parseInt(process.env.TOP_K_RESULTS || '5', 10),
    embeddingModel: process.env.EMBEDDING_MODEL || 'all-MiniLM-L6-v2',
  },
} as const;

// Create necessary directories
import fs from 'fs';
Object.values(config.paths).forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});