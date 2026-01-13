# VideoEmotion Administration System

Complete administration system for managing videos, trash, and pipeline execution.

## Quick Start

### 1. Install Dependencies

```bash
# Install API and dashboard dependencies
pip install -r requirements_api.txt

# Install main dependencies (if not already installed)
pip install -r requirements_main.txt
```

### 2. Start the API Server

```bash
# From project root
python app/api.py
```

The API will be available at `http://localhost:8000`

### 3. Start the Dashboard

```bash
# From project root
streamlit run app/dashboard_admin.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## Features

### 🎬 Administration
- View all videos (offline and realtime)
- Filter by mode and status
- Sort by name, date, or status
- Delete videos (moves to trash)
- View global statistics

### 🗑️ Trash
- View all trashed videos
- Restore videos from trash
- Permanently delete videos
- Empty entire trash
- See freed disk space

### 📥 Unprocessed Videos
- List videos that haven't been processed
- Quick access to pipeline runner
- See video metadata

### ⚙️ Pipeline Runner
- Configure pipeline options (FPS, smoothing, TTA, etc.)
- Select emotion detection backend
- Skip specific pipeline steps
- Monitor job progress in real-time
- View pipeline logs
- Cancel running jobs

### 📊 Analytics (Existing)
- View emotion analysis results
- Interactive charts and graphs
- Video playback with annotations
- Per-person emotion tracking

---

## API Endpoints

### Video Management
- `GET /api/videos` - List all videos
- `GET /api/videos/{id}` - Get video details
- `DELETE /api/videos/{id}` - Move to trash
- `GET /api/videos/unprocessed` - List unprocessed videos
- `POST /api/videos/scan` - Trigger video scan

### Trash Management
- `GET /api/trash` - List trash items
- `POST /api/trash/{id}/restore` - Restore from trash
- `DELETE /api/trash/{id}` - Permanently delete
- `POST /api/trash/empty` - Empty entire trash

### Pipeline Execution
- `POST /api/pipeline/run` - Start pipeline job
- `GET /api/pipeline/jobs/{id}` - Get job status
- `GET /api/pipeline/jobs` - List recent jobs
- `DELETE /api/pipeline/jobs/{id}` - Cancel job

### Statistics
- `GET /api/stats` - Get global statistics
- `POST /api/stats/refresh` - Force stats recalculation

### Health
- `GET /health` - Health check
- `GET /` - API info

---

## Architecture

### Backend Components

**Core Modules** (`src/core/`):
- `models.py` - Data models (VideoMetadata, TrashMetadata, PipelineJob, etc.)
- `video_manager.py` - Video inventory and scanning (async support)
- `trash_manager.py` - Trash operations with rollback (batch support)
- `stats_updater.py` - Statistics recalculation (async)
- `pipeline_executor.py` - Pipeline job management

**API** (`app/api.py`):
- FastAPI backend with all endpoints
- Background task support
- Async operations
- Error handling and validation

### Frontend Components

**Dashboard** (`app/dashboard_admin.py`):
- Unified interface for analytics and administration
- Page-based navigation
- API configuration

**Components** (`app/components/`):
- `admin_section.py` - Video management interface
- `trash_section.py` - Trash management interface
- `unprocessed_section.py` - Unprocessed videos list
- `pipeline_runner.py` - Pipeline configuration and execution

### Data Storage

**Metadata** (`video_metadata.json`):
- Video inventory
- Processing status
- File paths
- Statistics

**Trash** (`trash/`):
```
trash/
├── offline/
│   └── video_name_TIMESTAMP/
│       ├── .trash_metadata.json
│       └── [all related files]
└── realtime/
    └── session_TIMESTAMP/
        ├── .trash_metadata.json
        └── [session data]
```

**Pipeline Jobs** (`pipeline_jobs.json`):
- Job history
- Status and progress
- Logs and errors

---

## Configuration

### API Base URL

Configure in the dashboard sidebar under "API Configuration" or set in session state:

```python
st.session_state.api_base = "http://localhost:8000"
```

### Video Paths

Videos are expected in:
- **Offline**: `data/videos/`
- **Realtime**: `output/realtime/`

### Pipeline Configuration

Default pipeline options can be modified in the Pipeline Runner interface:
- FPS: 5
- Smoothing: Enabled
- TTA: Enabled
- Backend: HSEmotion
- Visualize: Enabled
- Export Bboxes: Enabled

---

## Performance Features

### Async Operations
- Video scanning runs asynchronously
- Statistics recalculation in background
- Pipeline execution doesn't block UI

### Batch Operations
- Batch video deletion
- Batch restoration from trash
- Batch permanent deletion

### Caching
- Video metadata cached in memory
- Statistics cached with TTL
- Lazy loading for large datasets

### Pagination
- 50 videos per page (configurable)
- Lazy loading of video details
- Efficient database queries

---

## Troubleshooting

### API Connection Failed
1. Make sure API server is running: `python app/api.py`
2. Check API base URL in dashboard settings
3. Verify firewall settings

### Videos Not Showing
1. Click "Refresh" button in Administration section
2. Trigger manual scan: `POST /api/videos/scan`
3. Check `video_metadata.json` file

### Pipeline Job Stuck
1. Check job logs in Pipeline Runner
2. Cancel job if needed
3. Check `pipeline_jobs.json` for errors
4. Verify video file exists in `data/videos/`

### Statistics Not Updating
1. Trigger manual refresh: `POST /api/stats/refresh`
2. Check that summary scripts exist:
   - `src/offline/emotion_summary_report.py`
   - `src/realtime/summarize_master.py`

---

## Development

### Running in Development Mode

```bash
# API with auto-reload
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

# Dashboard with auto-reload
streamlit run app/dashboard_admin.py
```

### Testing

```bash
# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/videos
curl http://localhost:8000/api/stats

# Test video scan
curl -X POST http://localhost:8000/api/videos/scan
```

### Logs

API logs are printed to console. For production, configure logging:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
```

---

## Security Considerations

### Current Implementation
- No authentication (local use only)
- CORS allows all origins
- No rate limiting

### Production Recommendations
1. Add authentication (JWT, OAuth)
2. Restrict CORS origins
3. Add rate limiting
4. Use HTTPS
5. Validate file paths strictly
6. Add audit logging
7. Implement user permissions

---

## Future Enhancements

- [ ] User authentication and authorization
- [ ] Video upload via dashboard
- [ ] Batch pipeline execution
- [ ] Email notifications for job completion
- [ ] Advanced search and filtering
- [ ] Export reports as PDF
- [ ] Video comparison view
- [ ] Real-time dashboard updates (WebSocket)
- [ ] Database backend (SQLite/PostgreSQL)
- [ ] Docker deployment
- [ ] Cloud storage integration

---

## License

VideoEmotion Administration System
Part of the VideoEmotion project
