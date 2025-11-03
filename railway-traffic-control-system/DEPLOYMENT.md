# Deployment Guide - Railway Traffic Control System

## Deployment Options

### Option 1: Local Development

**Step 1: Setup Python Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Step 2: Train Models**
```bash
# Ensure dataset is in root directory
python train_ml_models.py
```

**Step 3: Start Backend API**
```bash
cd backend/api
python app.py
```

**Step 4: Serve Frontend**
```bash
cd frontend
python -m http.server 8080
```

**Step 5: Access**
- Dashboard: http://localhost:8080/dashboard.html
- API: http://localhost:5000

---

### Option 2: Docker Deployment

**Step 1: Build Containers**
```bash
docker-compose build
```

**Step 2: Start Services**
```bash
docker-compose up -d
```

**Step 3: Verify**
```bash
docker-compose ps
docker-compose logs -f
```

**Step 4: Access**
- Dashboard: http://localhost
- API: http://localhost:5000

**Stop Services**
```bash
docker-compose down
```

---

### Option 3: Production Deployment (Ubuntu/Debian)

**Step 1: Install Dependencies**
```bash
sudo apt update
sudo apt install python3-pip python3-venv nginx -y
```

**Step 2: Setup Application**
```bash
cd /opt
git clone <repository> railway-system
cd railway-system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Step 3: Train Models**
```bash
python train_ml_models.py
```

**Step 4: Create Systemd Service**
```bash
sudo nano /etc/systemd/system/railway-api.service
```

**Service File Content:**
```ini
[Unit]
Description=Railway Traffic Control API
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/opt/railway-system
Environment="PATH=/opt/railway-system/venv/bin"
ExecStart=/opt/railway-system/venv/bin/gunicorn --workers 4 --bind 0.0.0.0:5000 backend.api.app:app

[Install]
WantedBy=multi-user.target
```

**Step 5: Configure Nginx**
```bash
sudo nano /etc/nginx/sites-available/railway-system
```

**Nginx Config:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /opt/railway-system/frontend;
        index dashboard.html;
        try_files $uri $uri/ =404;
    }

    # API Proxy
    location /api/ {
        proxy_pass http://localhost:5000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

**Step 6: Enable and Start**
```bash
sudo ln -s /etc/nginx/sites-available/railway-system /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable railway-api
sudo systemctl start railway-api
```

**Step 7: Verify**
```bash
sudo systemctl status railway-api
curl http://localhost:5000/api/health
```

---

## Environment Variables

Create `.env` file:
```bash
FLASK_ENV=production
FLASK_APP=backend/api/app.py
API_HOST=0.0.0.0
API_PORT=5000
MODEL_PATH=backend/models
LOG_LEVEL=INFO
```

---

## Monitoring

**Check API Logs:**
```bash
# Docker
docker-compose logs -f api

# Systemd
sudo journalctl -u railway-api -f
```

**Check System Health:**
```bash
curl http://localhost:5000/api/health
```

---

## Backup & Maintenance

**Backup Models:**
```bash
tar -czf models-backup-$(date +%Y%m%d).tar.gz backend/models/
```

**Update Models:**
```bash
# Retrain with new data
python train_ml_models.py

# Restart API
sudo systemctl restart railway-api  # or docker-compose restart api
```

---

## Troubleshooting

**Issue: Models not loading**
```bash
# Check model files exist
ls -lh backend/models/

# Verify permissions
chmod 644 backend/models/*.pkl
```

**Issue: API not responding**
```bash
# Check if port is in use
sudo netstat -tlnp | grep 5000

# Check firewall
sudo ufw allow 5000
```

**Issue: Frontend can't connect to API**
- Ensure CORS is enabled in Flask app
- Check API_BASE_URL in kpi_panel.js
- Verify network connectivity

---

## Security Recommendations

1. **Use HTTPS** in production (Let's Encrypt)
2. **Set strong passwords** for any database connections
3. **Restrict API access** using API keys or JWT tokens
4. **Regular updates** of dependencies
5. **Monitor logs** for suspicious activity

---

## Performance Tuning

**Gunicorn Workers:**
```bash
# Formula: (2 x CPU cores) + 1
gunicorn --workers 9 --bind 0.0.0.0:5000 app:app
```

**Nginx Optimization:**
```nginx
client_max_body_size 10M;
proxy_connect_timeout 300s;
proxy_send_timeout 300s;
proxy_read_timeout 300s;
```

---

## Scaling

**Horizontal Scaling:**
- Deploy multiple API instances
- Use load balancer (HAProxy, Nginx)
- Share model files via NFS or S3

**Vertical Scaling:**
- Increase worker processes
- Allocate more RAM/CPU
- Use GPU for model inference (if applicable)

---

**For questions, refer to README.md or contact the development team.**
