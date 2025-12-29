# Clipppy Web Application Setup

This guide covers setting up the multi-tenant web dashboard for Clipppy.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-web.txt
```

### 2. Configure Environment Variables

Add these to your `.env` file:

```bash
# Required for Twitch OAuth login
TWITCH_CLIENT_ID=your_twitch_client_id
TWITCH_CLIENT_SECRET=your_twitch_client_secret
TWITCH_REDIRECT_URI=http://localhost:8000/auth/twitch/callback

# Optional: YouTube OAuth (for posting)
YOUTUBE_CLIENT_ID=your_youtube_client_id
YOUTUBE_CLIENT_SECRET=your_youtube_client_secret
YOUTUBE_REDIRECT_URI=http://localhost:8000/auth/youtube/callback

# Security (generate with: python -c "import secrets; print(secrets.token_hex(32))")
SECRET_KEY=your-random-secret-key-here

# Admin users (comma-separated Twitch emails)
ADMIN_EMAILS=your-email@example.com
```

### 3. Set Up Twitch Application

1. Go to https://dev.twitch.tv/console/apps
2. Create a new application
3. Set OAuth Redirect URL to: `http://localhost:8000/auth/twitch/callback`
4. Copy Client ID and Client Secret to your `.env`

### 4. Start the Web Server

```bash
python start_web.py
```

Then open http://localhost:8000 in your browser.

### 5. (Optional) Start with Orchestrator

To also start the stream monitoring service:

```bash
python start_web.py --all
```

## Architecture

```
clipppy-2/
├── web/                    # Web application
│   ├── main.py             # FastAPI app
│   ├── models.py           # Database models
│   ├── routes/             # API routes
│   │   ├── auth.py         # Login/logout
│   │   ├── clips.py        # Clip management
│   │   ├── settings.py     # User settings
│   │   └── admin.py        # Admin dashboard
│   └── templates/          # HTML templates
│
├── services/               # Background services
│   ├── orchestrator.py     # Manages listeners for all users
│   ├── listener.py         # Per-user stream listener
│   └── tasks.py            # Celery tasks for clip processing
│
└── start_web.py            # Startup script
```

## User Flow

1. User clicks "Login with Twitch" → Twitch OAuth
2. User is redirected to Dashboard
3. User connects YouTube account (optional)
4. User configures settings (threshold, cooldown, etc.)
5. When user goes live on Twitch:
   - Orchestrator detects they're live
   - Starts a listener for their channel
   - Detects viral moments using ViralDetector
   - Creates clips and queues for enhancement
6. Clips appear in dashboard for review
7. User approves/edits → Posts to YouTube

## Admin Access

Users with emails listed in `ADMIN_EMAILS` get admin access:
- View all users
- View user details and settings
- Enable/disable accounts
- Grant/revoke admin access
- Impersonate users for debugging

## Database

SQLite database stored at `data/clipppy.db`. Tables:
- `users` - User accounts (Twitch auth, YouTube tokens)
- `user_settings` - User preferences
- `clips` - Clip records and status
- `stream_sessions` - Stream history

## API Endpoints

### Auth
- `GET /auth/twitch` - Start Twitch login
- `GET /auth/twitch/callback` - Twitch OAuth callback
- `GET /auth/youtube` - Connect YouTube
- `GET /auth/youtube/callback` - YouTube OAuth callback
- `GET /auth/logout` - Logout
- `GET /auth/me` - Get current user

### Clips
- `GET /api/clips` - List clips
- `GET /api/clips/pending` - Pending review clips
- `GET /api/clips/stats` - Clip statistics
- `GET /api/clips/{id}` - Get clip details
- `PATCH /api/clips/{id}` - Update clip
- `POST /api/clips/{id}/approve` - Approve and post
- `POST /api/clips/{id}/reject` - Reject clip
- `POST /api/clips/{id}/reenhance` - Re-enhance with new settings

### Settings
- `GET /api/settings` - Get settings
- `PATCH /api/settings` - Update settings
- `POST /api/settings/reset` - Reset to defaults

### Admin
- `GET /api/admin/stats` - System stats
- `GET /api/admin/users` - List users
- `GET /api/admin/users/{id}` - User details
- `POST /api/admin/users/{id}/toggle-active` - Enable/disable user
- `POST /api/admin/users/{id}/toggle-admin` - Grant/revoke admin
- `GET /api/admin/users/{id}/impersonate` - Get impersonation token

## Next Steps

1. **Add Stripe billing** - For subscription tiers
2. **Add email notifications** - When clips are ready
3. **Add WebSocket** - Real-time clip notifications
4. **Add S3 storage** - For clip files
5. **Add proper logging** - Centralized log aggregation
6. **Add Celery/Redis** - For production task queue
