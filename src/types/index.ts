export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: Date;
}

export interface QueryRequest {
  query: string;
  conversationHistory?: ChatMessage[];
}

export interface DocumentMetadata {
  source: string;
  page?: number;
  chunk?: number;
  type: string;
  uploadedAt?: Date;
}

export interface SearchResult {
  text: string;
  metadata: DocumentMetadata;
  score?: number;
}

export interface QueryResponse {
  answer: string;
  sources: SearchResult[];
  processingTime?: number;
}

export interface UploadResponse {
  success: boolean;
  message: string;
  filename: string;
  chunksCreated: number;
}

export interface KnowledgeBaseItem {
  text: string;
  metadata: DocumentMetadata;
  id: string;
}

export interface PythonServiceRequest {
  action: 'embed' | 'generate' | 'process_document';
  data: any;
}

export interface PythonServiceResponse {
  success: boolean;
  data?: any;
  error?: string;
}