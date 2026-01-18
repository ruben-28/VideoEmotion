# VideoEmotion Frontend

Modern Next.js-based web interface for the VideoEmotion video analysis platform. This application provides a comprehensive dashboard for managing videos, running emotion analysis pipelines, and viewing analytics.

## Prerequisites

Before running the frontend, ensure you have:

- **Node.js** (v18 or higher)
- **npm**, **yarn**, **pnpm**, or **bun** package manager
- **Backend API** running on `http://localhost:8000`

## Backend Setup

The frontend requires the VideoEmotion API to be running. Start the backend first:

```bash
# From the project root directory
cd ..
python app/main.py
```

The API will start on `http://localhost:8000`. Verify it's running by visiting:
- `http://localhost:8000` - API status
- `http://localhost:8000/health` - Health check

## Installation

Install dependencies:

```bash
npm install
# or
yarn install
# or
pnpm install
```

## Running the Frontend

Start the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

The application will be available at **[http://localhost:3000](http://localhost:3000)**.

## Available Interfaces

The frontend provides three main interfaces:

### 1. **Video Administration Dashboard** (`/`)
**URL:** `http://localhost:3000`

The main dashboard for managing your video library:

- **Video Management**: View, filter, and manage all videos
- **Batch Operations**: Select multiple videos for bulk processing or deletion
- **Pipeline Execution**: Run emotion analysis on videos with configurable options
- **Trash Management**: Soft-delete videos and restore them when needed
- **Real-time Stats**: Monitor total videos, processing status, and storage usage

**Features:**
- Filter by status: All, Processed, Partial, Unprocessed
- Select and batch process multiple videos
- Configure pipeline options (frame extraction, face detection, emotion analysis)
- Move videos to trash or permanently delete them
- Restore videos from trash

### 2. **Analytics Dashboard** (`/analytics`)
**URL:** `http://localhost:3000/analytics`

View detailed analytics and insights from processed videos:

- **Emotion Distribution**: Visualize emotion patterns across videos
- **Processing Statistics**: Track pipeline performance and completion rates
- **Temporal Analysis**: Analyze emotion trends over time
- **Video Comparisons**: Compare emotion profiles across different videos

### 3. **Realtime Analysis** (`/realtime`)
**URL:** `http://localhost:3000/realtime`

Run live emotion analysis on webcam or video streams:

- **Live Camera Feed**: Analyze emotions in real-time from your webcam
- **Stream Configuration**: Adjust detection sensitivity and frame rates
- **Instant Results**: See emotion predictions as they happen
- **Session Management**: Start, stop, and monitor realtime sessions

## Development

### Project Structure

```
frontend/
├── app/                    # Next.js app directory
│   ├── page.tsx           # Main dashboard (/)
│   ├── analytics/         # Analytics interface
│   ├── realtime/          # Realtime analysis interface
│   └── layout.tsx         # Root layout
├── components/            # Reusable UI components
├── public/               # Static assets
└── styles/               # Global styles
```

### Auto-Reload

The development server supports hot module replacement. Changes to any `.tsx`, `.ts`, or `.css` files will automatically reload in the browser.

### API Configuration

The frontend connects to the backend API at `http://localhost:8000` by default. To change this, update the `API_BASE` constant in the relevant page files.

## Building for Production

Create an optimized production build:

```bash
npm run build
npm start
```

The production server will run on `http://localhost:3000`.

## Troubleshooting

### "Could not connect to VideoEmotion API"

**Solution:** Ensure the backend API is running on port 8000:
```bash
cd ..
python app/main.py
```

### Port 3000 Already in Use

**Solution:** Kill the process using port 3000 or specify a different port:
```bash
PORT=3001 npm run dev
```

### Videos Not Displaying

**Solution:** Verify that:
1. The backend API is running
2. Videos exist in the `videos/` directory
3. The API has scanned the videos folder (check backend logs)

## Learn More

- [Next.js Documentation](https://nextjs.org/docs) - Learn about Next.js features and API
- [VideoEmotion API Docs](http://localhost:8000/docs) - Interactive API documentation (when backend is running)
- [React Documentation](https://react.dev) - Learn about React

## Technology Stack

- **Framework:** Next.js 15 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **UI Components:** Custom components with Radix UI primitives
- **API Communication:** Fetch API with REST endpoints
