import { Request, Response } from 'express';
import { vectorStore } from '../services/vectorStore';
import { pythonService } from '../services/pythonService';
import { QueryRequest, QueryResponse } from '../types';
import { config } from '../config';

export class ChatController {
  async query(req: Request, res: Response): Promise<void> {
    const startTime = Date.now();
    
    try {
      const { query, conversationHistory }: QueryRequest = req.body;

      if (!query || typeof query !== 'string') {
        res.status(400).json({ error: 'Query is required and must be a string' });
        return;
      }

      // Retrieve relevant documents from vector store
      const searchResults = await vectorStore.query(
        query, 
        config.vectorStore.topKResults
      );

      if (searchResults.length === 0) {
        const response: QueryResponse = {
          answer: "I don't have any information in my knowledge base yet. Please load the knowledge base or upload some documents first.",
          sources: [],
          processingTime: Date.now() - startTime,
        };
        res.json(response);
        return;
      }

      // Generate response using Python service (Ollama)
      const contextDocs = searchResults.map(result => result.text);
      const answer = await pythonService.generateResponse(query, contextDocs);

      const response: QueryResponse = {
        answer,
        sources: searchResults.slice(0, 3), // Return top 3 sources
        processingTime: Date.now() - startTime,
      };

      res.json(response);
    } catch (error) {
      console.error('Error in query controller:', error);
      res.status(500).json({ 
        error: 'Failed to process query',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  async healthCheck(req: Request, res: Response): Promise<void> {
    try {
      const pythonHealthy = await pythonService.healthCheck();
      const stats = await vectorStore.getCollectionStats();

      res.json({
        status: 'healthy',
        services: {
          python: pythonHealthy ? 'healthy' : 'unhealthy',
          vectorStore: 'healthy',
        },
        vectorStore: {
          documentsCount: stats.count,
        },
        model: config.ollama.model,
      });
    } catch (error) {
      res.status(503).json({ 
        status: 'unhealthy',
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }
}

export const chatController = new ChatController();