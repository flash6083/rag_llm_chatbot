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
    // Embedded ChromaDB, no HTTP server needed
    this.client = new ChromaClient({
      persistDirectory: path.resolve(__dirname, '../../chroma_db'),
    } as any);
    this.initializeCollection();
  }

  private async initializeCollection() {
    try {
      // Initially create or fetch collection
      this.collection = await this.client.getOrCreateCollection({
        name: this.collectionName,
        metadata: { 'hnsw:space': 'cosine' },
        embeddingFunction: null as any, // embeddings provided externally
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

  // Always get fresh collection to avoid stale references
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

  async loadJsonKnowledgeBase(jsonPath: string): Promise<number> {
    try {
      if (!fs.existsSync(jsonPath)) {
        throw new Error(`Knowledge base file not found: ${jsonPath}`);
      }

      const data = JSON.parse(await fsPromises.readFile(jsonPath, 'utf-8'));
      const items = this.flattenJson(data);

      if (!items.length) throw new Error('No items extracted from JSON');

      await this.addDocuments(items);
      return items.length;
    } catch (error) {
      console.error('Error loading JSON knowledge base:', error);
      throw error;
    }
  }

  private flattenJson(
    data: any,
    parentKey = '',
    items: KnowledgeBaseItem[] = []
  ): KnowledgeBaseItem[] {
    if (typeof data === 'object' && data !== null) {
      if (Array.isArray(data)) {
        data.forEach((item, index) =>
          this.flattenJson(item, `${parentKey}[${index}]`, items)
        );
      } else {
        Object.entries(data).forEach(([key, value]) => {
          const newKey = parentKey ? `${parentKey}.${key}` : key;
          if (typeof value === 'object' && value !== null) {
            const jsonStr = JSON.stringify(value, null, 2);
            items.push({
              id: uuidv4(),
              text: `${key}: ${jsonStr}`,
              metadata: { source: 'knowledge_base', type: 'json', category: parentKey || 'root' } as any,
            });
            this.flattenJson(value, newKey, items);
          } else {
            items.push({
              id: uuidv4(),
              text: `${key}: ${value}`,
              metadata: { source: 'knowledge_base', type: 'json', category: parentKey || 'root' } as any,
            });
          }
        });
      }
    }
    return items;
  }

  async clearCollection(): Promise<void> {
    try {
      await this.client.deleteCollection({ name: this.collectionName });
      // Recreate collection and get fresh reference
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
