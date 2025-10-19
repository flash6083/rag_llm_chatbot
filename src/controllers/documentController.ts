import { Request, Response } from 'express';
import path from 'path';
import fs from 'fs/promises';
import { vectorStore } from '../services/vectorStore';
import { pythonService } from '../services/pythonService';
import { UploadResponse, KnowledgeBaseItem } from '../types';
import { config } from '../config';
import { v4 as uuidv4 } from 'uuid';

export class DocumentController {
  async uploadDocument(req: Request, res: Response): Promise<void> {
    try {
      if (!req.file) {
        res.status(400).json({ error: 'No file uploaded' });
        return;
      }

      const file = req.file;
      const fileExtension = path.extname(file.originalname).toLowerCase();
      const supportedFormats = ['.pdf', '.docx', '.txt', '.json'];

      if (!supportedFormats.includes(fileExtension)) {
        res.status(400).json({ 
          error: 'Unsupported file format',
          supportedFormats 
        });
        return;
      }

      // Process document using Python service
      const chunks = await pythonService.processDocument(
        file.path,
        fileExtension.substring(1)
      );

      // Convert chunks to KnowledgeBaseItems
      const items: KnowledgeBaseItem[] = chunks.map((chunk: any, index: number) => ({
        id: `${file.filename}_${index}`,
        text: chunk.text,
        metadata: {
          source: file.originalname,
          page: chunk.page,
          chunk: index,
          type: fileExtension.substring(1),
          uploadedAt: new Date(),
        },
      }));

      // Add to vector store
      await vectorStore.addDocuments(items);

      const response: UploadResponse = {
        success: true,
        message: 'Document uploaded and processed successfully',
        filename: file.originalname,
        chunksCreated: items.length,
      };

      res.json(response);
    } catch (error) {
      console.error('Error uploading document:', error);
      res.status(500).json({ 
        error: 'Failed to process document',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  async loadKnowledgeBase(req: Request, res: Response): Promise<void> {
    try {
      const jsonPath = path.join(
        config.paths.knowledgeBaseDir,
        'department_data.json'
      );

      // Check if file exists
      try {
        await fs.access(jsonPath);
      } catch {
        res.status(404).json({ 
          error: 'Knowledge base file not found',
          expectedPath: jsonPath 
        });
        return;
      }

      const documentsLoaded = await vectorStore.loadJsonKnowledgeBase(jsonPath);

      res.json({
        success: true,
        message: 'Knowledge base loaded successfully',
        documentsLoaded,
      });
    } catch (error) {
      console.error('Error loading knowledge base:', error);
      res.status(500).json({ 
        error: 'Failed to load knowledge base',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  async clearKnowledgeBase(req: Request, res: Response): Promise<void> {
    try {
      await vectorStore.clearCollection();
      res.json({
        success: true,
        message: 'Knowledge base cleared successfully',
      });
    } catch (error) {
      console.error('Error clearing knowledge base:', error);
      res.status(500).json({ 
        error: 'Failed to clear knowledge base',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  async getStats(req: Request, res: Response): Promise<void> {
    try {
      const stats = await vectorStore.getCollectionStats();
      res.json({
        success: true,
        stats,
      });
    } catch (error) {
      console.error('Error getting stats:', error);
      res.status(500).json({ 
        error: 'Failed to get stats',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }
}

export const documentController = new DocumentController();