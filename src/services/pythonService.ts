import axios from 'axios';
import { config } from '../config';
import { PythonServiceRequest, PythonServiceResponse } from '../types';

class PythonService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = config.pythonServiceUrl;
  }

  /**
   * Generate embeddings for multiple texts
   * @param texts Array of texts to embed
   * @param normalize Whether to normalize embeddings (default: true for cosine similarity)
   */
  async embedTexts(texts: string[], normalize: boolean = true): Promise<number[][]> {
    try {
      const response = await axios.post<PythonServiceResponse>(
        `${this.baseUrl}/embed`,
        { texts, normalize },
        { timeout: 30000 }
      );

      if (!response.data.success) {
        throw new Error(response.data.error || 'Embedding failed');
      }

      return response.data.data.embeddings;
    } catch (error) {
      console.error('Error embedding texts:', error);
      throw error;
    }
  }

  /**
   * Embed a single query with optional expansion
   * @param query The search query
   * @param expand Whether to expand query with synonyms (default: true)
   */
  async embedQuery(query: string, expand: boolean = true): Promise<{
    embedding: number[];
    original_query: string;
    expanded_query: string | null;
    dimensions: number;
  }> {
    try {
      const response = await axios.post<PythonServiceResponse>(
        `${this.baseUrl}/embed-query`,
        { query, expand },
        { timeout: 10000 }
      );

      if (!response.data.success) {
        throw new Error(response.data.error || 'Query embedding failed');
      }

      return response.data.data;
    } catch (error) {
      console.error('Error embedding query:', error);
      throw error;
    }
  }

  /**
   * Re-rank search results using hybrid scoring
   * @param query Original user query
   * @param results Initial search results from ChromaDB
   * @param topK Number of top results to return
   */
  async rerank(
    query: string,
    results: Array<{ text: string; score?: number; metadata?: any }>,
    topK: number = 5
  ): Promise<Array<{
    text: string;
    score: number;
    metadata: any;
    final_score: number;
    score_breakdown: {
      vector: number;
      keyword: number;
      exact_match: number;
      length: number;
      position: number;
    };
  }>> {
    try {
      const response = await axios.post<PythonServiceResponse>(
        `${this.baseUrl}/rerank`,
        { query, results, top_k: topK },
        { timeout: 10000 }
      );

      if (!response.data.success) {
        throw new Error(response.data.error || 'Re-ranking failed');
      }

      return response.data.data.results;
    } catch (error) {
      console.error('Error re-ranking results:', error);
      throw error;
    }
  }

  /**
   * Analyze query to determine optimal retrieval parameters
   * @param query User query to analyze
   */
  async analyzeQuery(query: string): Promise<{
    query_type: string;
    suggested_params: {
      top_k: number;
      expand_query: boolean;
    };
    query_length: number;
    complexity: string;
  }> {
    try {
      const response = await axios.post<PythonServiceResponse>(
        `${this.baseUrl}/analyze-query`,
        { query },
        { timeout: 5000 }
      );

      if (!response.data.success) {
        throw new Error(response.data.error || 'Query analysis failed');
      }

      return response.data.data;
    } catch (error) {
      console.error('Error analyzing query:', error);
      // Fallback to defaults if analysis fails
      return {
        query_type: 'general',
        suggested_params: { top_k: 5, expand_query: true },
        query_length: query.split(' ').length,
        complexity: 'medium'
      };
    }
  }

  /**
   * Generate response using Ollama LLM (Enhanced version)
   * @param query User query
   * @param contextDocs Array of context documents with scores
   * @param includeSources Whether to include source information in response
   */
  async generate(
    query: string,
    contextDocs: Array<{ text: string; metadata?: any; final_score?: number; score?: number }>,
    includeSources: boolean = false
  ): Promise<{
    response: string;
    model_used: string;
    context_chunks: number;
    sources?: Array<{
      text: string;
      score: number;
      metadata: any;
    }>;
  }> {
    try {
      const response = await axios.post<PythonServiceResponse>(
        `${this.baseUrl}/generate`,
        {
          query,
          context_docs: contextDocs,
          include_sources: includeSources,
          model: config.ollama.model
        },
        { timeout: 60000 }
      );

      if (!response.data.success) {
        throw new Error(response.data.error || 'Generation failed');
      }

      return response.data.data;
    } catch (error) {
      console.error('Error generating response:', error);
      throw error;
    }
  }

  /**
   * Generate response using Ollama LLM (Legacy version for backward compatibility)
   * @param query User query
   * @param contextDocs Array of context document texts
   */
  async generateResponse(query: string, contextDocs: string[]): Promise<string> {
    try {
      const response = await axios.post<PythonServiceResponse>(
        `${this.baseUrl}/generate`,
        { 
          query, 
          context_docs: contextDocs.map(text => ({ text })),
          model: config.ollama.model 
        },
        { timeout: 60000 }
      );

      if (!response.data.success) {
        throw new Error(response.data.error || 'Generation failed');
      }

      return response.data.data.response;
    } catch (error) {
      console.error('Error generating response:', error);
      throw error;
    }
  }

  /**
   * Process document (PDF, DOCX, TXT) and extract chunks
   * @param filePath Path to the document file
   * @param fileType Type of file (pdf, docx, txt)
   */
  async processDocument(filePath: string, fileType: string): Promise<any[]> {
    try {
      const response = await axios.post<PythonServiceResponse>(
        `${this.baseUrl}/process-document`,
        { 
          file_path: filePath,
          file_type: fileType,
          chunk_size: config.vectorStore.chunkSize,
          chunk_overlap: config.vectorStore.chunkOverlap
        },
        { timeout: 120000 }
      );

      if (!response.data.success) {
        throw new Error(response.data.error || 'Document processing failed');
      }

      return response.data.data.chunks;
    } catch (error) {
      console.error('Error processing document:', error);
      throw error;
    }
  }

  /**
   * Health check for Python service
   */
  async healthCheck(): Promise<any> {
    try {
      const response = await axios.get(`${this.baseUrl}/health`, { timeout: 5000 });
      return response.data;
    } catch (error) {
      console.error('Python service health check failed:', error);
      return { status: 'unhealthy', error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  /**
   * Legacy health check (returns boolean)
   */
  async isHealthy(): Promise<boolean> {
    try {
      const health = await this.healthCheck();
      return health.status === 'healthy';
    } catch (error) {
      return false;
    }
  }
}

export const pythonService = new PythonService();