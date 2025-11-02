import { ChromaClient } from 'chromadb';
import { KnowledgeBaseItem, SearchResult } from '../types';
import { pythonService } from './pythonService';
import { v4 as uuidv4 } from 'uuid';
import fs from 'fs';
import path from 'path';
import fsPromises from 'fs/promises';

class VectorStoreService {
  private client: ChromaClient;
  private collectionName = 'nit_rourkela_cs_dept';
  private collection: any;
  private isInitialized = false;

  constructor() {
    this.client = new ChromaClient({
      persistDirectory: path.resolve(__dirname, '../../chroma_db'),
    } as any);
    this.initializeCollection();
  }

  private async initializeCollection() {
    try {
      this.collection = await this.client.getOrCreateCollection({
        name: this.collectionName,
        metadata: { 'hnsw:space': 'cosine' },
        embeddingFunction: null as any,
      });
      this.isInitialized = true;
      console.log('✅ ChromaDB collection initialized (embedded)');
    } catch (error) {
      console.error('❌ Error initializing ChromaDB collection:', error);
      throw error;
    }
  }

  private async ensureInitialized() {
    if (!this.isInitialized) {
      await this.initializeCollection();
    }
  }

  private async getCollection() {
    await this.ensureInitialized();
    this.collection = await this.client.getOrCreateCollection({
      name: this.collectionName,
      metadata: { 'hnsw:space': 'cosine' },
      embeddingFunction: null as any,
    });
    return this.collection;
  }

  async addDocuments(items: KnowledgeBaseItem[]): Promise<void> {
    try {
      if (items.length === 0) return;
      const collection = await this.getCollection();

      const texts = items.map(item => item.text);
      const embeddings = await pythonService.embedTexts(texts);

      await collection.add({
        ids: items.map(item => item.id),
        embeddings,
        documents: texts,
        metadatas: items.map(item => item.metadata),
      });

      console.log(`✅ Added ${items.length} documents to vector store`);
    } catch (error) {
      console.error('Error adding documents:', error);
      throw error;
    }
  }

  async query(queryText: string, nResults: number = 5): Promise<SearchResult[]> {
    try {
      const collection = await this.getCollection();
      const [queryEmbedding] = await pythonService.embedTexts([queryText]);

      const results = await collection.query({
        queryEmbeddings: [queryEmbedding],
        nResults,
      });

      if (!results.documents?.[0]?.length) return [];

      return results.documents[0].map((doc: string, i: number) => ({
        text: doc,
        metadata: results.metadatas[0][i],
        score: results.distances?.[0]?.[i],
      }));
    } catch (error) {
      console.error('Error querying vector store:', error);
      throw error;
    }
  }

  async queryWithEmbedding(
    embedding: number[],
    nResults: number = 5
  ): Promise<SearchResult[]> {
    try {
      const collection = await this.getCollection();

      const results = await collection.query({
        queryEmbeddings: [embedding],
        nResults,
      });

      if (!results.documents?.[0]?.length) return [];

      return results.documents[0].map((doc: string, i: number) => ({
        text: doc,
        metadata: results.metadatas[0][i],
        score: results.distances?.[0]?.[i] || 0,
      }));
    } catch (error) {
      console.error('Error querying with embedding:', error);
      throw error;
    }
  }

  async loadJsonKnowledgeBase(jsonPath: string): Promise<number> {
    try {
      if (!fs.existsSync(jsonPath)) {
        throw new Error(`Knowledge base file not found: ${jsonPath}`);
      }

      const data = JSON.parse(await fsPromises.readFile(jsonPath, 'utf-8'));
      const items = this.smartFlattenJson(data);

      if (!items.length) throw new Error('No items extracted from JSON');

      console.log(`📦 Created ${items.length} knowledge base chunks`);
      await this.addDocuments(items);
      return items.length;
    } catch (error) {
      console.error('Error loading JSON knowledge base:', error);
      throw error;
    }
  }

  /**
   * Improved JSON flattening that keeps related data together
   * This ensures faculty names stay with their courses, research areas, etc.
   */
  private smartFlattenJson(data: any): KnowledgeBaseItem[] {
    const items: KnowledgeBaseItem[] = [];

    // Handle top-level arrays (e.g., faculty list, courses list)
    if (Array.isArray(data)) {
      data.forEach((item, index) => {
        if (typeof item === 'object' && item !== null) {
          // Create a comprehensive text representation
          const text = this.objectToText(item);
          
          items.push({
            id: uuidv4(),
            text,
            metadata: {
              source: 'knowledge_base',
              type: 'json',
              category: 'root',
              index,
            } as any,
          });
        }
      });
    } 
    // Handle top-level objects with named sections
    else if (typeof data === 'object' && data !== null) {
      Object.entries(data).forEach(([sectionName, sectionData]) => {
        if (Array.isArray(sectionData)) {
          // Process each item in the array (e.g., each faculty member)
          sectionData.forEach((item, index) => {
            if (typeof item === 'object' && item !== null) {
              const text = this.objectToText(item, sectionName);
              
              items.push({
                id: uuidv4(),
                text,
                metadata: {
                  source: 'knowledge_base',
                  type: 'json',
                  category: sectionName,
                  index,
                } as any,
              });
            } else {
              // Simple value
              items.push({
                id: uuidv4(),
                text: `${sectionName}: ${item}`,
                metadata: {
                  source: 'knowledge_base',
                  type: 'json',
                  category: sectionName,
                  index,
                } as any,
              });
            }
          });
        } else if (typeof sectionData === 'object' && sectionData !== null) {
          // Process object sections
          const text = this.objectToText(sectionData, sectionName);
          
          items.push({
            id: uuidv4(),
            text,
            metadata: {
              source: 'knowledge_base',
              type: 'json',
              category: sectionName,
            } as any,
          });
        } else {
          // Simple key-value pair
          items.push({
            id: uuidv4(),
            text: `${sectionName}: ${sectionData}`,
            metadata: {
              source: 'knowledge_base',
              type: 'json',
              category: 'root',
            } as any,
          });
        }
      });
    }

    return items;
  }

  /**
   * Convert an object to a comprehensive text representation
   * Keeps all related fields together (name, email, courses, research, etc.)
   */
  private objectToText(obj: any, prefix?: string): string {
    const lines: string[] = [];
    
    if (prefix) {
      lines.push(`[${prefix}]`);
    }

    const formatValue = (value: any, indent: string = ''): string => {
      if (Array.isArray(value)) {
        return value.map(v => `${indent}- ${v}`).join('\n');
      } else if (typeof value === 'object' && value !== null) {
        return Object.entries(value)
          .map(([k, v]) => `${indent}${k}: ${formatValue(v, indent + '  ')}`)
          .join('\n');
      }
      return String(value);
    };

    // Process all fields
    Object.entries(obj).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        if (Array.isArray(value)) {
          lines.push(`${key}:`);
          lines.push(formatValue(value, '  '));
        } else if (typeof value === 'object') {
          lines.push(`${key}:`);
          lines.push(formatValue(value, '  '));
        } else {
          lines.push(`${key}: ${value}`);
        }
      }
    });

    return lines.join('\n');
  }

  async clearCollection(): Promise<void> {
    try {
      await this.client.deleteCollection({ name: this.collectionName });
      await this.getCollection();
      console.log('✅ Collection cleared and re-initialized');
    } catch (error) {
      console.error('Error clearing collection:', error);
      throw error;
    }
  }

  async getCollectionStats(): Promise<{ count: number }> {
    try {
      const collection = await this.getCollection();
      const count = await collection.count();
      return { count };
    } catch (error) {
      console.error('Error getting collection stats:', error);
      return { count: 0 };
    }
  }
}

export const vectorStore = new VectorStoreService();