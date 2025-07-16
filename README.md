# AI News Synopsis Flask App

A Flask application that fetches news from various sources, analyzes them using AI, and verifies their authenticity using async/await for optimal performance.

## Features

- **Async/await support**: All external API calls are asynchronous for better performance
- **News fetching**: Retrieves news from configured URLs (TMZ, BBC)
- **AI analysis**: Uses DeepSeek AI to extract news stories from HTML content
- **News verification**: Uses Dify AI to verify news authenticity and provide summaries
- **Concurrent processing**: Processes multiple URLs and news items simultaneously
- **RESTful API**: Simple POST endpoint for triggering analysis
- **Retry mechanism**: Automatic retry (up to 3 times) for failed verifications
- **Deduplication**: Prevents duplicate verification of the same news titles

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API keys in `app.py`:
   - Replace `DEEPSEEK_API_KEY` with your actual DeepSeek API key
   - Replace `DIFY_API_KEY` with your actual Dify API key

## Usage

### Running the App

```bash
python app.py
```

The app will start on `http://localhost:5000`

### API Endpoints

#### Analyze News
```
POST /analyze
```
Triggers news extraction and verification from all configured sources.

**Query Parameters:**
- `force_refresh=true`: Bypass cache and force fresh analysis

**Response Format:**
```json
{
  "data": [
    {
      "title": "News title",
      "summary": "News summary", 
      "source_link": "https://example.com/article",
      "happened_today": "yes"
    }
  ],
  "cached": true,
  "execution_time": 45.23,
  "cache_info": {
    "cache_age_hours": 2.5,
    "hours_remaining": 21.5,
    "execution_time": 45.23,
    "results_count": 4,
    "is_valid": true
  }
}
```

#### Analyze News (Force Refresh)
```
POST /analyze-force
```
Forces a fresh analysis, bypassing cache completely.

#### Analyze (Wait for Refresh)
```
POST /analyze-wait-refresh
```
Waits for background refresh to complete and returns updated cache content.
If no refresh is in progress, returns current cache content.

**Response Format:**
```json
{
  "data": [...],
  "cached": true,
  "cache_info": {...},
  "execution_time": 45.23,
  "message": "Background refresh completed, returning updated cache"
}
```

#### Refresh Cache (Background)
```
POST /refresh-cache
```
Starts background cache refresh. Returns immediately with status.

**Response Format:**
```json
{
  "status": "started",
  "message": "Background cache refresh started",
  "refresh_status": {
    "refresh_in_progress": true,
    "elapsed_seconds": 0,
    "start_time": "2024-01-01T12:00:00"
  }
}
```

#### Refresh Status
```
GET /refresh-status
```
Returns the current status of background cache refresh.

**Response Format:**
```json
{
  "refresh_status": {
    "refresh_in_progress": true,
    "elapsed_seconds": 45.23,
    "start_time": "2024-01-01T12:00:00"
  },
  "cache_info": {
    "cache_age_hours": 2.5,
    "hours_remaining": 21.5,
    "execution_time": 45.23,
    "results_count": 4,
    "is_valid": true
  }
}
```

#### Cache Information
```
GET /cache-info
```
Returns information about the current cache status.

**Response Format:**
```json
{
  "cache_exists": true,
  "cache_info": {
    "cache_age_hours": 2.5,
    "hours_remaining": 21.5,
    "execution_time": 45.23,
    "results_count": 4,
    "is_valid": true
  },
  "cache_file_path": "cache/analyze_cache.json",
  "cache_duration_hours": 24
}
```

#### Clear Cache
```
POST /clear-cache
```
Clears the cache file.

**Response Format:**
```json
{
  "status": "success",
  "message": "Cache cleared successfully"
}
```

#### Debug Endpoint
```
POST /analyze-debug
```
Returns fixed debug content in the same format as `/analyze` for testing purposes.

#### Generate Image
```
POST /generate-image
```
Generates an image based on news title and summary using DALL-E 3.

**Request Body:**
```json
{
  "title": "News title",
  "summary": "News summary"
}
```

**Response:**
```json
{
  "status": "success",
  "image_url": "https://oaidalleapiprodscus.blob.core.windows.net/...",
  "prompt": "Generated prompt for image creation",
  "title": "News title",
  "summary": "News summary"
}
```

#### Generate Image (Debug)
```
POST /generate-image-debug
```
Returns a mock image URL for testing without API calls.

**Request Body:**
```json
{
  "title": "Test News Title",
  "summary": "Test news summary"
}
```

**Response:**
```json
{
  "status": "success",
  "image_url": "https://via.placeholder.com/1024x1024/0066cc/ffffff?text=Generated+Image",
  "prompt": "Generated prompt for image creation",
  "title": "Test News Title",
  "summary": "Test news summary",
  "debug": true
}
```

## Example Usage

### Trigger news analysis (with cache):
```bash
curl -X POST http://localhost:5000/analyze
```

### Force fresh analysis (bypass cache):
```bash
# Method 1: Using query parameter
curl -X POST "http://localhost:5000/analyze?force_refresh=true"

# Method 2: Using dedicated endpoint
curl -X POST http://localhost:5000/analyze-force
```

### Background cache refresh:
```bash
# Start background refresh
curl -X POST http://localhost:5000/refresh-cache

# Check refresh status
curl http://localhost:5000/refresh-status

# Wait for refresh to complete and get updated content
curl -X POST http://localhost:5000/analyze-wait-refresh

# Check cache info
curl http://localhost:5000/cache-info
```

### Clear cache:
```bash
curl -X POST http://localhost:5000/clear-cache
```

### Test with debug endpoint:
```bash
curl -X POST http://localhost:5000/analyze-debug
```

### Generate image from news:
```bash
# Generate real image (requires OpenAI API key)
curl -X POST http://localhost:5000/generate-image \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Breakthrough in Solar Panel Technology",
    "summary": "Scientists have developed a new solar panel design that increases efficiency by 25%"
  }'

# Test with debug endpoint (no API key required)
curl -X POST http://localhost:5000/generate-image-debug \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test News Title",
    "summary": "Test news summary"
  }'
```

### Using Python requests:
```python
import requests

response = requests.post('http://localhost:5000/analyze')
results = response.json()
print(results)
```

## Configuration

You can modify the following variables in `app.py`:

- `target_urls`: List of news sources to fetch from (currently TMZ, BBC)
- `query_prefix`: The prompt used for news verification
- `DEEPSEEK_API_KEY`: Your DeepSeek API key
- `DIFY_API_KEY`: Your Dify API key
- `OPENAI_API_KEY`: Your OpenAI API key (for image generation)

## Deployment

### Local Development

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd ai-synopsis-flask
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set environment variables:**
```bash
export DEEPSEEK_API_KEY="your-deepseek-key"
export DIFY_API_KEY="your-dify-key"
```

5. **Run the application:**
```bash
python app.py
```

### Production Deployment

#### Option 1: Using Gunicorn (Recommended)

1. **Install Gunicorn:**
```bash
pip install gunicorn
```

2. **Create a WSGI entry point** (create `wsgi.py`):
```python
from app import app

if __name__ == "__main__":
    app.run()
```

3. **Run with Gunicorn:**
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 wsgi:app
```

4. **For production, use a process manager like systemd:**
```bash
# Create service file: /etc/systemd/system/ai-synopsis.service
[Unit]
Description=AI Synopsis Flask App
After=network.target

[Service]
User=your-user
WorkingDirectory=/path/to/ai-synopsis-flask
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn --bind 0.0.0.0:5000 --workers 4 wsgi:app
Restart=always

[Install]
WantedBy=multi-user.target
```

#### Option 2: Using Docker

1. **Create Dockerfile:**
```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
```

2. **Build and run:**
```bash
docker build -t ai-synopsis .
docker run -p 5000:5000 -e DEEPSEEK_API_KEY=your-key -e DIFY_API_KEY=your-key ai-synopsis
```

3. **Using Docker Compose** (create `docker-compose.yml`):
```yaml
version: '3.8'
services:
  ai-synopsis:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - DIFY_API_KEY=${DIFY_API_KEY}
    restart: unless-stopped
```

#### Option 3: Cloud Deployment

##### Heroku

1. **Create `Procfile`:**
```
web: gunicorn --bind 0.0.0.0:$PORT --workers 4 app:app
```

2. **Deploy:**
```bash
heroku create your-app-name
heroku config:set DEEPSEEK_API_KEY=your-key
heroku config:set DIFY_API_KEY=your-key
git push heroku main
```

##### AWS Elastic Beanstalk

1. **Create `.ebextensions/01_environment.config`:**
```yaml
option_settings:
  aws:elasticbeanstalk:application:environment:
    DEEPSEEK_API_KEY: your-key
    DIFY_API_KEY: your-key
```

2. **Deploy:**
```bash
eb init
eb create production
eb deploy
```

##### Google Cloud Run

1. **Create `cloudbuild.yaml`:**
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/ai-synopsis', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/ai-synopsis']
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'ai-synopsis'
      - '--image'
      - 'gcr.io/$PROJECT_ID/ai-synopsis'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
```

2. **Deploy:**
```bash
gcloud builds submit --config cloudbuild.yaml
```

### Environment Variables

For production deployment, set these environment variables:

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export DIFY_API_KEY="your-dify-api-key"
export FLASK_ENV="production"
export FLASK_DEBUG="false"
```

### Performance Considerations

1. **Worker Processes**: Use 2-4 workers for most deployments
2. **Timeout Settings**: Set appropriate timeouts for long-running AI operations
3. **Rate Limiting**: Consider implementing rate limiting for the `/analyze` endpoint
4. **Caching**: Consider caching results to avoid repeated API calls
5. **Monitoring**: Set up logging and monitoring for production deployments

### Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **HTTPS**: Always use HTTPS in production
3. **Input Validation**: Validate all inputs to prevent injection attacks
4. **Rate Limiting**: Implement rate limiting to prevent abuse
5. **CORS**: Configure CORS appropriately for your frontend

## Architecture

The app uses:
- **Flask**: Web framework
- **httpx**: Async HTTP client for fetching HTML content
- **aiohttp**: Async HTTP client for Dify API calls
- **BeautifulSoup**: HTML parsing
- **AsyncOpenAI**: Async OpenAI client for DeepSeek API
- **asyncio**: Python's async/await support for concurrent processing

## How It Works

1. **HTML Fetching**: Concurrently fetches HTML content from all configured URLs using httpx
2. **AI Analysis**: Uses DeepSeek AI to extract news titles from HTML content
3. **News Verification**: For each news title, uses Dify AI to verify authenticity and get summaries
4. **Response**: Returns a JSON array of verified news items with metadata

## Performance

- All external API calls are asynchronous
- Multiple URLs are processed concurrently
- News verification is limited to the first 2 items per source to avoid rate limits
- HTML content is truncated to 60,000 characters to stay within token limits
- Automatic retry mechanism for failed verifications

## Error Handling

- Graceful handling of failed HTML fetches
- Fallback for failed AI analysis
- JSON parsing error handling for verification results
- Timeout handling for all external requests
- Retry mechanism for failed verifications (up to 3 attempts)

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your DeepSeek and Dify API keys are valid and have sufficient quota
2. **Timeout Errors**: Increase timeout values for slow API responses
3. **Memory Issues**: Reduce the number of concurrent requests or worker processes
4. **Rate Limiting**: Implement delays between requests if hitting API rate limits

### Debug Mode

Use the debug endpoint to test without external APIs:
```bash
curl -X POST http://localhost:5000/analyze-debug
```

### Logs

Check application logs for detailed error information:
```bash
# If using systemd
journalctl -u ai-synopsis -f

# If using Docker
docker logs <container-id>
``` 