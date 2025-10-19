import { Router, Request, Response, NextFunction } from 'express';
import { chatController } from '../controllers/chatController';
import { documentController } from '../controllers/documentController';
import { upload } from '../middleware/upload';

const router = Router();

// Async handler wrapper (for extra safety, though Express 5 handles this)
const asyncHandler = (fn: (req: Request, res: Response, next: NextFunction) => Promise<void>) => {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};

// Chat routes
router.post('/query', asyncHandler(async (req, res, next) => {
  await chatController.query(req, res);
}));

router.get('/health', asyncHandler(async (req, res, next) => {
  await chatController.healthCheck(req, res);
}));

// Document routes
router.post('/upload', upload.single('file'), asyncHandler(async (req, res, next) => {
  await documentController.uploadDocument(req, res);
}));

router.post('/load-knowledge-base', asyncHandler(async (req, res, next) => {
  await documentController.loadKnowledgeBase(req, res);
}));

router.delete('/knowledge-base', asyncHandler(async (req, res, next) => {
  await documentController.clearKnowledgeBase(req, res);
}));

router.get('/stats', asyncHandler(async (req, res, next) => {
  await documentController.getStats(req, res);
}));

export default router;