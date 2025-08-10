# 🚀 Quick Start Guide

Get your Law RAG Chatbot running in 5 minutes!

## ⚡ Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
# Windows
set HF_TOKEN=your_huggingface_token
set GROQ_API_KEY=your_groq_api_key

# Linux/Mac
export HF_TOKEN=your_huggingface_token
export GROQ_API_KEY=your_groq_api_key
```

### 3. Run the API
```bash
python run_api.py
```

### 4. Test the API
```bash
python test_client.py
```

## 🔑 Get API Keys

- **Hugging Face**: https://huggingface.co/settings/tokens
- **Groq**: https://console.groq.com/

## 🌐 Access Points

- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

## 💡 Example Usage

```bash
# Chat with the bot
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are tenant rights?", "context_length": 3}'

# Search documents
curl "http://localhost:8000/search?query=eviction&limit=5"
```

## 🆘 Need Help?

- Run `python setup_env.py` for interactive setup
- Check the full [README.md](README.md) for detailed instructions
- Use `python demo.py` to see the system in action

---

**That's it! Your RAG chatbot is ready to use! 🎉** 