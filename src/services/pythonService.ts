import axios from 'axios';
import { config } from '../config';
import { PythonServiceRequest, PythonServiceResponse } from '../types';

class PythonService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = config.pythonServiceUrl;
  }

  async embedTexts(texts: string[]): Promise<number[][]> {
    try {
      const response = await axios.post<PythonServiceResponse>(
        `${this.baseUrl}/embed`,
        { texts },
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

  async generateResponse(query: string, contextDocs: string[]): Promise<string> {
    try {
      const response = await axios.post<PythonServiceResponse>(
        `${this.baseUrl}/generate`,
        { 
          query, 
          context_docs: contextDocs,
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

  async healthCheck(): Promise<boolean> {
    try {
      const response = await axios.get(`${this.baseUrl}/health`, { timeout: 5000 });
      return response.data.status === 'healthy';
    } catch (error) {
      return false;
    }
  }
}

export const pythonService = new PythonService();