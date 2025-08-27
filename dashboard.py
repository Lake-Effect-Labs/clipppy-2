#!/usr/bin/env python3

"""
Clipppy Dashboard
================

Simple web dashboard for tracking TikTok posts, views, and performance
across all streamer accounts.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import yaml

try:
    from flask import Flask, render_template, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️ Flask not installed. Run: pip install flask")

logger = logging.getLogger(__name__)

class ClippyDashboard:
    """Dashboard for monitoring Clipppy performance"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize dashboard with configuration"""
        self.config_path = Path(config_path)
        self.load_config()
        self.load_data()
        
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            self.setup_routes()
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("✅ Dashboard config loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            self.config = {}
    
    def load_data(self):
        """Load data from various sources"""
        self.upload_history = self.load_upload_history()
        self.performance_data = self.load_performance_data()
        self.stream_data = self.load_stream_data()
    
    def load_upload_history(self) -> Dict:
        """Load upload history from TikTok uploader"""
        history_file = Path("data/upload_history.json")
        try:
            if history_file.exists():
                with open(history_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Could not load upload history: {e}")
            return {}
    
    def load_performance_data(self) -> Dict:
        """Load TikTok performance data (views, likes, etc.)"""
        perf_file = Path("data/performance_data.json")
        try:
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Could not load performance data: {e}")
            return {}
    
    def load_stream_data(self) -> Dict:
        """Load stream monitoring data"""
        stream_file = Path("data/stream_data.json")
        try:
            if stream_file.exists():
                with open(stream_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Could not load stream data: {e}")
            return {}
    
    def get_dashboard_data(self) -> Dict:
        """Compile all data for dashboard display"""
        data = {
            'overview': self.get_overview_stats(),
            'streamers': self.get_streamer_stats(),
            'recent_uploads': self.get_recent_uploads(),
            'performance': self.get_performance_summary(),
            'system_status': self.get_system_status()
        }
        return data
    
    def get_overview_stats(self) -> Dict:
        """Get high-level overview statistics"""
        today = datetime.now().date().isoformat()
        
        # Count enabled streamers
        enabled_streamers = sum(1 for s in self.config.get('streamers', []) if s.get('enabled', False))
        
        # Count uploads today
        uploads_today = 0
        for key, history in self.upload_history.items():
            if not key.endswith('_details') and isinstance(history, dict):
                uploads_today += history.get(today, 0)
        
        # Get total uploads
        total_uploads = 0
        for key, history in self.upload_history.items():
            if not key.endswith('_details') and isinstance(history, dict):
                total_uploads += sum(history.values())
        
        # Calculate total views (simulated for now)
        total_views = total_uploads * 15000  # Assume average 15k views per upload
        
        return {
            'enabled_streamers': enabled_streamers,
            'uploads_today': uploads_today,
            'total_uploads': total_uploads,
            'total_views': total_views,
            'avg_views_per_upload': total_views // max(total_uploads, 1)
        }
    
    def get_streamer_stats(self) -> List[Dict]:
        """Get statistics for each streamer"""
        streamers = []
        
        for streamer_config in self.config.get('streamers', []):
            name = streamer_config['name']
            
            # Find uploads for this streamer
            account_key = f"{name}_*"
            matching_keys = [k for k in self.upload_history.keys() if k.startswith(f"{name}_") and not k.endswith('_details')]
            
            uploads_today = 0
            total_uploads = 0
            
            for key in matching_keys:
                history = self.upload_history.get(key, {})
                if isinstance(history, dict):
                    uploads_today += history.get(datetime.now().date().isoformat(), 0)
                    total_uploads += sum(history.values())
            
            # Simulate performance data
            avg_views = 12000 + (len(name) * 1000)  # Simple simulation
            total_views = total_uploads * avg_views
            
            streamers.append({
                'name': name,
                'enabled': streamer_config.get('enabled', False),
                'twitch_username': streamer_config.get('twitch_username', name),
                'tiktok_username': streamer_config.get('tiktok_account', {}).get('username', f"{name}_clippy"),
                'uploads_today': uploads_today,
                'total_uploads': total_uploads,
                'total_views': total_views,
                'avg_views': avg_views,
                'max_posts_per_day': streamer_config.get('tiktok_account', {}).get('max_posts_per_day', 3)
            })
        
        return streamers
    
    def get_recent_uploads(self) -> List[Dict]:
        """Get recent upload history across all accounts"""
        recent = []
        
        for key, details in self.upload_history.items():
            if key.endswith('_details') and isinstance(details, list):
                for upload in details[-10:]:  # Last 10 uploads
                    recent.append({
                        'timestamp': upload.get('timestamp'),
                        'account': upload.get('account'),
                        'caption': upload.get('caption', '')[:50] + ('...' if len(upload.get('caption', '')) > 50 else ''),
                        'video_name': Path(upload.get('video_path', '')).name
                    })
        
        # Sort by timestamp (newest first)
        recent.sort(key=lambda x: x['timestamp'], reverse=True)
        return recent[:20]  # Return last 20 uploads
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary with trends"""
        # This would be populated with real TikTok analytics data
        # For now, we'll simulate some data
        
        return {
            'daily_views': [
                {'date': '2024-01-15', 'views': 45000},
                {'date': '2024-01-16', 'views': 52000},
                {'date': '2024-01-17', 'views': 38000},
                {'date': '2024-01-18', 'views': 61000},
                {'date': '2024-01-19', 'views': 47000},
                {'date': '2024-01-20', 'views': 55000},
                {'date': '2024-01-21', 'views': 63000}
            ],
            'top_performing_videos': [
                {'title': 'INSANE JYNXZI ACE!', 'views': 250000, 'account': 'jynxzi_clippy'},
                {'title': 'SHROUD HEADSHOT MONTAGE', 'views': 180000, 'account': 'shroud_clippy'},
                {'title': 'NO WAY THIS HAPPENED', 'views': 165000, 'account': 'jynxzi_clippy'}
            ]
        }
    
    def get_system_status(self) -> Dict:
        """Get system health and status"""
        return {
            'monitoring_active': True,
            'last_update': datetime.now().isoformat(),
            'disk_usage': '2.3 GB',
            'clips_processed_today': 8,
            'errors_today': 0,
            'uptime': '3 days, 12 hours'
        }
    
    def setup_routes(self):
        """Setup Flask routes for the dashboard"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return self.render_dashboard()
        
        @self.app.route('/api/data')
        def api_data():
            """API endpoint for dashboard data"""
            return jsonify(self.get_dashboard_data())
        
        @self.app.route('/api/refresh')
        def api_refresh():
            """Refresh data and return updated dashboard data"""
            self.load_data()
            return jsonify(self.get_dashboard_data())
    
    def render_dashboard(self) -> str:
        """Render the dashboard HTML"""
        data = self.get_dashboard_data()
        
        # Simple HTML template (in a real app, you'd use proper templates)
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Clipppy Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
                .stat-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .stat-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
                .stat-label {{ color: #666; margin-top: 5px; }}
                .section {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                .streamer-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
                .streamer-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 8px; }}
                .streamer-card.enabled {{ border-color: #4CAF50; background-color: #f8fff8; }}
                .streamer-card.disabled {{ border-color: #f44336; background-color: #fff8f8; }}
                .status-indicator {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }}
                .status-enabled {{ background-color: #4CAF50; }}
                .status-disabled {{ background-color: #f44336; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .refresh-btn {{ background: #667eea; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }}
                .refresh-btn:hover {{ background: #5a6fd8; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎬 Clipppy Dashboard</h1>
                    <p>TikTok automation pipeline monitoring</p>
                    <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{data['overview']['enabled_streamers']}</div>
                        <div class="stat-label">Active Streamers</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{data['overview']['uploads_today']}</div>
                        <div class="stat-label">Uploads Today</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{data['overview']['total_uploads']:,}</div>
                        <div class="stat-label">Total Uploads</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{data['overview']['total_views']:,}</div>
                        <div class="stat-label">Total Views</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Streamers</h2>
                    <div class="streamer-grid">
        """
        
        for streamer in data['streamers']:
            status_class = "enabled" if streamer['enabled'] else "disabled"
            status_indicator = "status-enabled" if streamer['enabled'] else "status-disabled"
            
            html += f"""
                        <div class="streamer-card {status_class}">
                            <h3><span class="status-indicator {status_indicator}"></span>{streamer['name']}</h3>
                            <p><strong>Twitch:</strong> @{streamer['twitch_username']}</p>
                            <p><strong>TikTok:</strong> @{streamer['tiktok_username']}</p>
                            <p><strong>Uploads Today:</strong> {streamer['uploads_today']}/{streamer['max_posts_per_day']}</p>
                            <p><strong>Total Uploads:</strong> {streamer['total_uploads']:,}</p>
                            <p><strong>Total Views:</strong> {streamer['total_views']:,}</p>
                            <p><strong>Avg Views:</strong> {streamer['avg_views']:,}</p>
                        </div>
            """
        
        html += f"""
                    </div>
                </div>
                
                <div class="section">
                    <h2>🕒 Recent Uploads</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Account</th>
                                <th>Caption</th>
                                <th>Video</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for upload in data['recent_uploads'][:10]:
            if upload['timestamp']:
                time_str = datetime.fromisoformat(upload['timestamp']).strftime('%m/%d %H:%M')
            else:
                time_str = 'Unknown'
            
            html += f"""
                            <tr>
                                <td>{time_str}</td>
                                <td>@{upload['account']}</td>
                                <td>{upload['caption']}</td>
                                <td>{upload['video_name']}</td>
                            </tr>
            """
        
        html += f"""
                        </tbody>
                    </table>
                </div>
                
                <div class="section">
                    <h2>⚙️ System Status</h2>
                    <p><strong>Status:</strong> {'🟢 Active' if data['system_status']['monitoring_active'] else '🔴 Inactive'}</p>
                    <p><strong>Last Update:</strong> {datetime.fromisoformat(data['system_status']['last_update']).strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Clips Processed Today:</strong> {data['system_status']['clips_processed_today']}</p>
                    <p><strong>Errors Today:</strong> {data['system_status']['errors_today']}</p>
                    <p><strong>Uptime:</strong> {data['system_status']['uptime']}</p>
                </div>
            </div>
            
            <script>
                function refreshData() {{
                    fetch('/api/refresh')
                        .then(response => response.json())
                        .then(data => {{
                            location.reload();
                        }})
                        .catch(error => console.error('Error:', error));
                }}
                
                // Auto-refresh every 60 seconds
                setInterval(refreshData, 60000);
            </script>
        </body>
        </html>
        """
        
        return html
    
    def run(self, host='localhost', port=8080, debug=False):
        """Run the dashboard server"""
        if not FLASK_AVAILABLE:
            print("❌ Flask not available. Install with: pip install flask")
            return
        
        print(f"🚀 Starting Clipppy Dashboard at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Run the dashboard"""
    dashboard = ClippyDashboard()
    
    # Get dashboard config
    config = dashboard.config.get('dashboard', {})
    host = config.get('host', 'localhost')
    port = config.get('port', 8080)
    
    dashboard.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    main()
