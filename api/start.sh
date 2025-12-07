#!/bin/bash
# Quick start script for FastAPI with migrations

echo "🚀 Starting FastAPI with Database Migrations"
echo "=============================================="

# Build and start services
echo "📦 Building and starting services..."
docker-compose up --build -d sps-genai-postgres sps-genai-fast-api

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
sleep 10

# Run migrations
echo "🔄 Running database migrations..."
docker-compose exec sps-genai-fast-api python migrate.py migrate

echo "✅ Setup complete!"
echo ""
echo "🌐 Services available at:"
echo "   FastAPI: http://localhost:8888"
echo "   API Docs: http://localhost:8888/docs"
echo "   PostgreSQL: localhost:55432"
echo ""
echo "🔧 Useful commands:"
echo "   Check migration status: docker-compose exec sps-genai-fast-api python migrate.py status"
echo "   View logs: docker-compose logs sps-genai-fast-api"
echo "   Stop services: docker-compose down"
