# Environment Configuration Guide

## Quick Setup

1. **Copy the example file:**
   ```bash
   # For development
   cp env.example .env
   
   # For production
   cp env.production.example .env
   ```

2. **Edit `.env` and add your OpenAI API key:**
   ```bash
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. **Start the application:**
   ```bash
   python start.py
   ```

## Environment Variables

### Required Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required)

### Optional Variables

All other variables have sensible defaults. See the example files for details:

- **Development**: `env.example`
- **Production**: `env.production.example`

## Important Notes

1. **Never commit `.env` files** - They contain sensitive information
2. **Use different keys** for development and production
3. **CORS_ORIGINS** - Use comma-separated values: `http://localhost:3000,https://yourdomain.com`
4. **File sizes** - MAX_FILE_SIZE is in bytes (500MB = 524288000)

## Docker

When using Docker Compose, environment variables can be set in:
- `docker-compose.yml` (environment section)
- `.env` file (read automatically by docker-compose)

## Production Checklist

- [ ] Set `DEBUG=false`
- [ ] Update `CORS_ORIGINS` with production URLs
- [ ] Use PostgreSQL for `DATABASE_URL`
- [ ] Set `LOG_FORMAT=json`
- [ ] Configure appropriate `MAX_FILE_SIZE`
- [ ] Set `RATE_LIMIT_PER_MINUTE` based on capacity
- [ ] Store `OPENAI_API_KEY` securely (use secrets manager)

