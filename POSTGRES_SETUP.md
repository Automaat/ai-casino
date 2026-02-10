# PostgreSQL Production Setup

## Overview

Production daemon uses PostgreSQL for persistent storage with automatic migrations and 90-day data retention.

## Quick Start

```bash
# 1. Start services (postgres + daemon + dashboard)
docker-compose up -d

# 2. Check logs
docker-compose logs -f daemon

# 3. Verify database connection
docker-compose exec postgres psql -U ai_casino -d ai_casino -c "\dt"
```

## Data Persistence

PostgreSQL data is stored on your host machine at:
```
~/.ai-casino/postgres-data/
```

**This survives container restarts and docker-compose down** - data will not be lost unless you manually delete this directory.

## Configuration

All database settings are in `~/.ai-casino/daemon-production.yaml`:

```yaml
database:
  database_url: "postgresql+asyncpg://ai_casino:YOUR_PASSWORD_HERE@postgres:5432/ai_casino"
  pool_size: 5
  max_overflow: 10
  pool_pre_ping: true
  enable_persistence: true

state:
  cleanup_enabled: true
  cleanup_retention_days: 90
  cleanup_hour: 3
```

## Migrations

Migrations run automatically on daemon startup:
- `006_add_analysis_records.sql` - Trading analysis history
- `007_add_position_records.sql` - Active positions
- `008_add_position_actions.sql` - Position management actions
- `009_add_discovery_history.sql` - Stock discovery outcomes

## Backfill Historical Data

If you have existing JSON state, backfill to database:

```bash
# From inside daemon container (use password from your daemon-production.yaml)
docker-compose exec daemon python -m src.database.migrations.backfill_from_json \
  --state-file ~/.ai-casino/daemon-state.json \
  --database-url postgresql+asyncpg://ai_casino:YOUR_PASSWORD_HERE@postgres:5432/ai_casino
```

## Retention Policy

Automatic cleanup runs daily at 3 AM (configurable):
- Analysis records: 90 days
- Position actions: 90 days
- Discovery history: 90 days

## Accessing the Database

### Via Docker

```bash
# psql shell
docker-compose exec postgres psql -U ai_casino -d ai_casino

# Run query
docker-compose exec postgres psql -U ai_casino -d ai_casino \
  -c "SELECT symbol, signal, confidence, timestamp FROM analysis_records ORDER BY timestamp DESC LIMIT 10;"
```

### Via Host (if port exposed)

```bash
# Use password from your daemon-production.yaml or .env file
psql postgresql://ai_casino:YOUR_PASSWORD_HERE@localhost:5432/ai_casino
```

## Useful Queries

```sql
-- Analysis history by symbol
SELECT symbol, signal, confidence, timestamp
FROM analysis_records
WHERE symbol = 'AAPL'
ORDER BY timestamp DESC
LIMIT 20;

-- Active positions
SELECT symbol, entry_price, current_qty, days_held, current_stop_loss
FROM position_records;

-- Recent position actions
SELECT symbol, action_type, price, reason, timestamp
FROM position_management_actions
ORDER BY timestamp DESC
LIMIT 20;

-- Discovery success rate
SELECT added_to_watchlist, COUNT(*)
FROM discovery_history
GROUP BY added_to_watchlist;

-- Signal distribution
SELECT signal, COUNT(*)
FROM analysis_records
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY signal;
```

## Troubleshooting

### Database not connecting
```bash
# Check postgres is running
docker-compose ps postgres

# Check postgres logs
docker-compose logs postgres

# Verify health
docker-compose exec postgres pg_isready -U ai_casino
```

### Migrations not applied
```bash
# Check daemon logs for migration errors
docker-compose logs daemon | grep -i migration

# Manually check migrations table
docker-compose exec postgres psql -U ai_casino -d ai_casino \
  -c "SELECT * FROM schema_migrations ORDER BY applied_at DESC;"
```

### Reset database (DEV ONLY)
```bash
# Stop services
docker-compose down

# Delete postgres data (WARNING: ALL DATA LOST)
rm -rf ~/.ai-casino/postgres-data

# Restart (will create fresh database)
docker-compose up -d
```

## Backup

```bash
# Create backup
docker-compose exec postgres pg_dump -U ai_casino ai_casino > backup_$(date +%Y%m%d).sql

# Restore backup
cat backup_20250210.sql | docker-compose exec -T postgres psql -U ai_casino ai_casino
```
