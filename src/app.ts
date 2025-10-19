import express, { Express, Request, Response, NextFunction } from 'express';
import cors from 'cors';
// Remove this line: import 'express-async-errors';
import routes from './routes';

const app: Express = express();

// Middleware
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:5173'],
  credentials: true,
}));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Request logging
app.use((req: Request, res: Response, next: NextFunction) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

// Routes
app.get('/', (req: Request, res: Response) => {
  res.json({
    message: 'NIT Rourkela CS Department Chatbot API',
    version: '1.0.0',
    endpoints: {
      health: '/api/v1/health',
      query: '/api/v1/query',
      upload: '/api/v1/upload',
      loadKB: '/api/v1/load-knowledge-base',
    },
  });
});

app.use('/api/v1', routes);

// Error handling middleware
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  console.error('Error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message,
  });
});

// 404 handler
app.use((req: Request, res: Response) => {
  res.status(404).json({ error: 'Route not found' });
});

export default app;