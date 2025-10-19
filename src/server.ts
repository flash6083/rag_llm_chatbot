import app from './app';
import { config } from './config';

const PORT = config.port;

app.listen(PORT, () => {
  console.log('\n🚀 NIT Rourkela CS Department Chatbot');
  console.log(`📡 Server running on http://localhost:${PORT}`);
  console.log(`🔧 Environment: ${config.env}`);
  console.log(`🤖 Ollama Model: ${config.ollama.model}`);
  console.log(`🐍 Python Service: ${config.pythonServiceUrl}`);
  console.log('\n✅ Ready to serve requests!\n');
});