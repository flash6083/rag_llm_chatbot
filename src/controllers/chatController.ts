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

      console.log(`📥 Query: "${query}"`);

      // Step 1: Analyze query to get optimal parameters
      const queryAnalysis = await pythonService.analyzeQuery(query);
      const topK = queryAnalysis.suggested_params.top_k;
      const expandQuery = queryAnalysis.suggested_params.expand_query;
      
      console.log(`🔍 Query type: ${queryAnalysis.query_type}, fetching top ${topK} results`);

      // Step 2: Get embedding with optional expansion
      const { embedding, expanded_query } = await pythonService.embedQuery(query, expandQuery);
      
      if (expanded_query && expanded_query !== query) {
        console.log(`📝 Query expanded: "${query}" → "${expanded_query}"`);
      }

      // Step 3: Search in ChromaDB using the embedding
      // Fetch more results than needed for re-ranking
      const initialResults = await vectorStore.queryWithEmbedding(
        embedding,
        Math.min(topK * 2, 15) // Get 2x results for re-ranking
      );

      if (initialResults.length === 0) {
        const response: QueryResponse = {
          answer: "I don't have any information in my knowledge base yet. Please load the knowledge base or upload some documents first.",
          sources: [],
          processingTime: Date.now() - startTime,
        };
        res.json(response);
        return;
      }

      console.log(`🔎 Found ${initialResults.length} initial results`);

      // Step 4: Re-rank results using hybrid scoring (CRITICAL!)
      const rerankedResults = await pythonService.rerank(
        query,
        initialResults,
        topK
      );

      console.log(`✨ Re-ranked to top ${rerankedResults.length} results`);
      
      // Log top result score for debugging
      if (rerankedResults.length > 0) {
        console.log(`📊 Top result score: ${rerankedResults[0].final_score.toFixed(3)}`);
      }

      // Step 5: Generate response using Ollama with re-ranked context
      const generationResult = await pythonService.generate(
        query,
        rerankedResults,
        true // Include sources
      );

      const response: QueryResponse = {
        answer: generationResult.response,
        sources: rerankedResults.slice(0, 3).map(result => ({
          text: result.text,
          metadata: result.metadata,
          score: result.final_score,
          scoreBreakdown: result.score_breakdown // Include for debugging
        })),
        processingTime: Date.now() - startTime,
        metadata: {
          queryType: queryAnalysis.query_type,
          contextChunks: generationResult.context_chunks,
          expandedQuery: expanded_query !== query ? expanded_query : undefined,
          modelUsed: generationResult.model_used
        }
      };

      console.log(`✅ Response generated in ${response.processingTime}ms`);
      res.json(response);

    } catch (error) {
      console.error('❌ Error in query controller:', error);
      res.status(500).json({ 
        error: 'Failed to process query',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  async healthCheck(req: Request, res: Response): Promise<void> {
    try {
      const pythonHealth = await pythonService.healthCheck();
      const stats = await vectorStore.getCollectionStats();

      res.json({
        status: 'healthy',
        services: {
          python: pythonHealth.status === 'healthy' ? 'healthy' : 'unhealthy',
          vectorStore: 'healthy',
        },
        vectorStore: {
          documentsCount: stats.count,
        },
        python: {
          ollamaModel: pythonHealth.ollama_model,
          embeddingModel: pythonHealth.embedding_model,
          embeddingDimensions: pythonHealth.embedding_dimensions
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

  /**
   * Optional: Test endpoint to compare old vs new retrieval
   */
  async compareRetrieval(req: Request, res: Response): Promise<void> {
    try {
      const { query } = req.body;
      
      if (!query) {
        res.status(400).json({ error: 'Query required' });
        return;
      }

      // Old method: Direct text search
      const oldResults = await vectorStore.query(query, 5);

      // New method: Embedding + re-ranking
      const { embedding } = await pythonService.embedQuery(query, true);
      const newInitial = await vectorStore.queryWithEmbedding(embedding, 10);
      const newReranked = await pythonService.rerank(query, newInitial, 5);

      res.json({
        query,
        old_method: {
          count: oldResults.length,
          results: oldResults.map(r => ({
            text: r.text.substring(0, 100) + '...',
            score: r.score
          }))
        },
        new_method: {
          initial_count: newInitial.length,
          reranked_count: newReranked.length,
          results: newReranked.map(r => ({
            text: r.text.substring(0, 100) + '...',
            vector_score: r.score,
            final_score: r.final_score,
            breakdown: r.score_breakdown
          }))
        }
      });

    } catch (error) {
      res.status(500).json({
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }
}

export const chatController = new ChatController();