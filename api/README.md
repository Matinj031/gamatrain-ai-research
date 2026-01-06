# Gamatrain AI Server

🤖 سرور هوش مصنوعی آموزشی Gamatrain با قابلیت RAG، حافظه مکالمه و اجرا بدون GPU.

## ✨ قابلیت‌ها

### RAG (Retrieval-Augmented Generation)
- جستجوی هوشمند در 2000+ بلاگ و محتوای وبسایت
- استفاده از embedding model چندزبانه (`intfloat/multilingual-e5-large`)
- ذخیره‌سازی index برای سرعت بالاتر

### Conversation Memory
- ذخیره 5 پیام آخر هر session
- پشتیبانی از سوالات follow-up مثل "Can you explain more?"
- تشخیص خودکار سوالات مرتبط با مکالمه قبلی

### Production Ready
- اجرا بدون نیاز به GPU
- استفاده از Groq API (رایگان و سریع)
- Streaming response با انیمیشن تایپ

## 🏗️ معماری

```
Frontend (Nuxt) → Production Server (CPU) → Groq API
                         ↓
                  RAG + Memory + Follow-up
                  (LlamaIndex + Embeddings)
                         ↓
                  Gamatrain API (Blogs, Schools)
```

## 🚀 نصب و اجرا

### Development

```bash
cd api
pip install -r requirements-production.txt
cp .env.example .env
# Edit .env with your GROQ_API_KEY
python llm_server_production.py
```

### Docker

```bash
docker-compose -f docker-compose.production.yml up -d
```

## ⚙️ تنظیمات (.env)

```env
# Provider
PROVIDER=groq

# Groq API (FREE - https://console.groq.com)
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Server
HOST=0.0.0.0
PORT=8002

# RAG
SIMILARITY_THRESHOLD=0.45
MAX_TOKENS=1024

# Gamatrain API
GAMATRAIN_API_URL=https://185.204.170.142/api/v1
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/query` | ارسال سوال (با streaming) |
| `POST` | `/v1/chat/completions` | OpenAI-compatible endpoint |
| `POST` | `/v1/refresh` | بروزرسانی RAG index |
| `DELETE` | `/v1/session/{id}` | پاک کردن حافظه session |
| `GET` | `/health` | بررسی سلامت سرور |
| `GET` | `/v1/debug/search?q=...` | Debug: جستجو با score |
| `GET` | `/v1/debug/list-blogs?search=...` | Debug: لیست بلاگ‌ها |

### نمونه Request

```bash
# Query with streaming
curl -X POST "http://localhost:8002/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "session_id": "user123"}'

# Refresh index
curl -X POST "http://localhost:8002/v1/refresh"
```

## 📁 ساختار فایل‌ها

```
api/
├── llm_server_production.py  # سرور اصلی (بدون GPU)
├── llm_server.py             # سرور توسعه (با Ollama)
├── requirements-production.txt
├── .env
└── storage/                  # RAG index cache
```

## 📝 نکات مهم

1. **Refresh Index**: بعد از اضافه شدن بلاگ جدید:
   ```bash
   curl -X POST "http://localhost:8002/v1/refresh"
   ```

2. **Groq API**: رایگان است ولی rate limit دارد (30 req/min)

3. **Embedding Model**: اولین اجرا ~2GB دانلود میکند

4. **Session Management**: هر کاربر باید `session_id` یکتا داشته باشد

## 🔧 Troubleshooting

**مشکل: RAG محتوا پیدا نمیکند**
- Index را refresh کنید
- Threshold را کاهش دهید (پیشنهاد: 0.45)

**مشکل: پاسخ کوتاه است**
- `MAX_TOKENS` را افزایش دهید

**مشکل: خطای API**
- `GROQ_API_KEY` را بررسی کنید
- Rate limit را چک کنید

## 📄 License

MIT
