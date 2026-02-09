#!/bin/bash

# F1PA Deployment Script
# Simple script to deploy F1PA locally or to a remote server

set -e

echo "=================================="
echo "F1PA Deployment Script"
echo "=================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
echo "Checking dependencies..."
if ! command_exists docker; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose found"
echo ""

# Deployment type
echo "Select deployment type:"
echo "1) Local development"
echo "2) Production"
read -p "Enter choice [1-2]: " deployment_type

case $deployment_type in
    1)
        echo ""
        echo "🚀 Starting local deployment..."
        echo ""

        # Build images
        echo "Building Docker images..."
        docker compose build

        # Start services
        echo "Starting services..."
        docker compose up -d

        # Wait for services to be ready
        echo "Waiting for services to start..."
        sleep 10

        # Health check
        echo ""
        echo "Checking services health..."
        docker compose ps

        echo ""
        echo "Testing API health..."
        curl -s -u f1pa:f1pa http://localhost:8000/health | python -m json.tool || echo "⚠️  API not ready yet"

        echo ""
        echo "✅ Local deployment complete!"
        echo ""
        echo "Access your services:"
        echo "  - API Documentation: http://localhost:8000/docs"
        echo "  - Streamlit UI: http://localhost:8501"
        echo "  - MLflow: http://localhost:5000"
        echo "  - Grafana: http://localhost:3000 (admin/admin)"
        echo "  - Prometheus: http://localhost:9090"
        echo ""
        echo "View logs: docker compose logs -f"
        echo "Stop services: docker compose down"
        ;;

    2)
        echo ""
        echo "🚀 Starting production deployment..."
        echo ""

        read -p "Enter server SSH address (user@host): " ssh_address

        if [ -z "$ssh_address" ]; then
            echo "❌ SSH address is required"
            exit 1
        fi

        echo "Deploying to $ssh_address..."

        # Copy files to server
        echo "Copying project files..."
        ssh $ssh_address "mkdir -p ~/f1pa"
        rsync -avz --exclude='data' --exclude='models' --exclude='mlartifacts' --exclude='.git' \
              --exclude='__pycache__' --exclude='*.pyc' --exclude='venv' \
              . $ssh_address:~/f1pa/

        # Deploy on server
        echo "Deploying on server..."
        ssh $ssh_address << 'EOF'
            cd ~/f1pa

            echo "Building images..."
            docker compose build

            echo "Starting services..."
            docker compose up -d

            echo "Waiting for services..."
            sleep 15

            echo "Services status:"
            docker compose ps

            echo ""
            echo "✅ Production deployment complete!"
EOF

        echo ""
        echo "Deployment to $ssh_address finished!"
        echo "SSH to server and check: ssh $ssh_address 'cd ~/f1pa && docker compose ps'"
        ;;

    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
