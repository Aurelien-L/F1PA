# F1PA - Guide de développement et déploiement

Guide complet pour le développement, les tests, le CI/CD et le déploiement de F1PA.

---

## 📋 Table des matières

- [Installation](#-installation)
- [Développement local](#-développement-local)
- [Tests & Qualité](#-tests--qualité)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Déploiement](#-déploiement)
- [Monitoring](#-monitoring)
- [Maintenance](#-maintenance)

---

## 🔧 Installation

### Prérequis

- Python 3.10+
- Docker & Docker Compose
- Git

### Configuration environnement local

```bash
# Cloner le projet
git clone https://github.com/Aurelien-L/F1PA.git
cd F1PA

# Créer environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Installer dépendances
pip install --upgrade pip
pip install -r requirements.txt

# Installer outils de développement
pip install pytest pytest-cov pytest-asyncio pylint
```

### Démarrer les services

```bash
# Lancer tous les services Docker
docker compose up -d

# Vérifier l'état
docker compose ps

# Tester l'API
curl -u f1pa:f1pa http://localhost:8000/health
```

**Services disponibles** :
- API Documentation : http://localhost:8000/docs
- Streamlit UI : http://localhost:8501
- MLflow : http://localhost:5000
- Grafana : http://localhost:3000 (admin/admin)
- Prometheus : http://localhost:9090

---

## 💻 Développement local

### Pipeline ETL

```bash
# Pipeline complet (Extract → Transform → Load)
python scripts/etl_pipeline.py --years 2023 2024 2025

# Pipeline sans Extract (données déjà extraites)
python scripts/etl_pipeline.py --years 2023 2024 2025 --skip-extract

# Vérification qualité uniquement
python scripts/etl_pipeline.py --verify-only

# Extraction drivers standalone
python scripts/extract_drivers.py --years 2023 2024 2025
```

### Machine Learning

```bash
# Entraîner le modèle
python ml/run_ml_pipeline.py

# Générer rapport de drift
docker exec f1pa_api python scripts/generate_drift_report.py
```

### Docker - Commandes courantes

```bash
# Services
docker compose up -d              # Démarrer tous les services
docker compose down               # Arrêter tous les services
docker compose restart            # Redémarrer les services
docker compose logs -f            # Voir les logs en temps réel
docker compose logs -f api        # Logs d'un service spécifique

# Build
docker compose build              # Builder toutes les images
docker compose build api          # Builder une image spécifique
docker compose up -d --build      # Builder et démarrer

# Nettoyage
docker compose down -v            # Arrêter et supprimer les volumes
docker system prune -af           # Nettoyer tout (⚠️ ATTENTION)
```

### Astuces utiles

```bash
# Exécuter commandes dans container
docker exec -it f1pa_api bash
docker exec f1pa_api python scripts/generate_drift_report.py

# PostgreSQL
docker exec -it f1pa_postgres psql -U f1pa -d f1pa_db

# Backup PostgreSQL
docker exec f1pa_postgres pg_dump -U f1pa f1pa_db > backup.sql

# Restore PostgreSQL
cat backup.sql | docker exec -i f1pa_postgres psql -U f1pa f1pa_db
```

---

## 🧪 Tests & Qualité

### Vérification code avec Pylint

```bash
# Vérifier tous les modules
pylint --rcfile=pyproject.toml api/ ml/ etl/ monitoring/ streamlit/ tests/ scripts/

# Vérifier un module spécifique
pylint --rcfile=pyproject.toml api/

# Vérifier un fichier spécifique
pylint --rcfile=pyproject.toml api/main.py
```

**Score cible** : > 9.0/10

### Tests unitaires

```bash
# Lancer tous les tests (unit + integration)
pytest tests/ -v

# Lancer uniquement les tests unitaires (sans services Docker)
pytest tests/ -v -m "not integration"

# Lancer uniquement les tests d'intégration (nécessite docker compose up -d)
pytest tests/ -v -m "integration"

# Avec coverage
pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

# Lancer un test spécifique
pytest tests/test_api.py -v

# Lancer une fonction de test spécifique
pytest tests/test_api.py::test_health_endpoint -v
```

**Tests disponibles** :
- 29 tests unitaires (peuvent tourner sans services Docker)
- 11 tests d'intégration (nécessitent `docker compose up -d`)
- Total : 40 tests, 100% de pass

**Note** : Les tests d'intégration sont automatiquement exclus du pipeline GitHub Actions pour garder le CI/CD simple et rapide.

### Pipeline CI/CD local

Simuler le pipeline GitHub Actions en local avant de push :

```bash
# 1. Lint
pylint --rcfile=pyproject.toml api/ ml/ etl/ monitoring/ streamlit/ tests/ scripts/

# 2. Tests (uniquement tests unitaires comme en CI)
pytest tests/ -v --cov -m "not integration"

# 3. Tests complets (unit + integration, nécessite docker compose)
docker compose up -d
pytest tests/ -v --cov  # Tous les tests

# 4. Build Docker
docker compose build

# 5. Vérifier santé
docker compose ps
curl -u f1pa:f1pa http://localhost:8000/health
```

---

## 🚀 CI/CD Pipeline

### Architecture GitHub Actions

```
Push → Lint → Tests → Build → Deploy
       ↓      ↓       ↓
    pylint  pytest  docker
           29 unit   images
           tests
```

### Workflows automatiques

**`.github/workflows/ci.yml`** - Pipeline principal

Déclenché sur :
- Push sur `main`, `dev`, `feat-*`
- Pull requests vers `main`, `dev`

Étapes :
1. **Lint** : Vérification code avec pylint
2. **Tests** : Exécution pytest + coverage (29 tests unitaires, PostgreSQL service uniquement)
   - Tests d'intégration exclus pour garder le CI/CD simple et rapide
3. **Build** : Construction images Docker (uniquement sur main/dev)

**`.github/workflows/release.yml`** - Workflow de release

Déclenché sur :
- Tags `v*.*.*`

Crée automatiquement une release GitHub avec les images Docker versionnées.

### Créer une release

```bash
# Créer et pusher un tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# Le workflow release.yml se déclenche automatiquement
```

### Activer le déploiement automatique

Pour activer le déploiement automatique vers la production, décommenter la section `deploy` dans `.github/workflows/ci.yml` et configurer les secrets GitHub :

**Secrets à configurer** (Settings → Secrets → Actions) :
- `SSH_PRIVATE_KEY` : Clé SSH privée pour le serveur
- `SERVER_HOST` : IP ou domaine du serveur
- `SERVER_USER` : Utilisateur SSH

---

## 📦 Déploiement

### Déploiement automatisé (script interactif)

```bash
bash scripts/deploy.sh
```

Options :
1. **Local development** : Build + démarrage services
2. **Production** : Déploiement via SSH sur serveur distant

### Déploiement manuel en production

#### 1. Préparation serveur

```bash
# Connexion SSH
ssh user@server

# Installation Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Installation Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Vérification
docker --version
docker compose version
```

**Configuration minimale serveur** :
- 4GB RAM
- 20GB disque
- Ports : 5000, 5432, 8000, 8501, 3000, 9090

#### 2. Cloner et configurer

```bash
# Cloner le repository
git clone https://github.com/Aurelien-L/F1PA.git
cd F1PA

# Checkout version spécifique (optionnel)
git checkout v1.0.0

# Configuration (optionnel)
cp .env.example .env
nano .env
```

**Variables d'environnement importantes** :
```bash
API_USERNAME=f1pa
API_PASSWORD=YOUR_SECURE_PASSWORD
POSTGRES_PASSWORD=YOUR_SECURE_PASSWORD
GF_SECURITY_ADMIN_PASSWORD=YOUR_SECURE_PASSWORD
```

#### 3. Lancement

```bash
# Build des images
docker compose build

# Lancement en mode détaché
docker compose up -d

# Vérification
docker compose ps
docker compose logs -f
```

#### 4. Vérification santé

```bash
# Health check API
curl -u f1pa:f1pa http://localhost:8000/health

# Test prédiction
curl -u f1pa:f1pa -X POST http://localhost:8000/predict/lap \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "driver_number": 1,
      "circuit_key": 7,
      "st_speed": 310,
      "i1_speed": 295,
      "i2_speed": 285,
      "temp": 25,
      "rhum": 60,
      "pres": 1013,
      "lap_number": 10,
      "year": 2024,
      "circuit_avg_laptime": 85.5,
      "driver_avg_laptime": 84.2,
      "driver_perf_score": 0.85
    }
  }'
```

### Mise à jour production

#### Rolling update (sans downtime)

```bash
# Pull dernières modifications
git pull origin main

# Rebuild seulement ce qui a changé
docker compose build

# Restart avec rolling update
docker compose up -d --no-deps --build api
docker compose up -d --no-deps --build streamlit

# Vérification
docker compose ps
```

#### Update complet

```bash
# Stop tous les services
docker compose down

# Pull + rebuild
git pull origin main
docker compose build

# Redémarrage
docker compose up -d

# Clean des anciennes images
docker image prune -f
```

---

## 📊 Monitoring

### Métriques Prometheus

```bash
# Vérifier les targets
curl http://localhost:9090/api/v1/targets | python -m json.tool

# Query métriques
curl "http://localhost:9090/api/v1/query?query=f1pa_predictions_total"
```

### Grafana Dashboards

1. Accéder à http://localhost:3000
2. Login : admin / admin
3. Dashboard : F1PA → F1PA ML Model Monitoring

**Panels disponibles** :
- Prediction Requests/sec
- Prediction Latency (p95)
- Model Status
- Error Rate
- Database/MLflow Connection Status

### Evidently - Drift detection

```bash
# Générer rapport de drift
docker exec f1pa_api python scripts/generate_drift_report.py

# Rapport HTML généré dans:
# monitoring/evidently/reports/test_data_drift.html
```

### Logs

```bash
# Logs temps réel
docker compose logs -f

# Logs spécifiques
docker compose logs -f api
docker compose logs -f streamlit

# Logs avec timestamp
docker compose logs -f --timestamps

# Dernières 100 lignes
docker compose logs --tail=100 api
```

---

## 🛠️ Maintenance

### Troubleshooting

#### Service ne démarre pas

```bash
# Vérifier les logs
docker compose logs service_name

# Vérifier l'état
docker compose ps

# Restart service spécifique
docker compose restart service_name
```

#### Problème de connexion base de données

```bash
# Vérifier que PostgreSQL est UP
docker compose ps postgres

# Tester la connexion
docker exec -it f1pa_postgres psql -U f1pa -d f1pa_db -c "SELECT 1;"

# Recréer la base (⚠️ perte de données)
docker compose down -v
docker compose up -d postgres
```

#### Manque d'espace disque

```bash
# Nettoyer les images inutilisées
docker system prune -a

# Nettoyer les volumes (⚠️ perte de données)
docker volume prune

# Vérifier l'espace
docker system df
```

#### Performance dégradée

```bash
# Vérifier ressources
docker stats

# Limiter ressources dans docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

### Backup & Restore

#### Backup base de données

```bash
# Backup manuel
docker exec f1pa_postgres pg_dump -U f1pa f1pa_db > backup_$(date +%Y%m%d).sql

# Backup automatique (cron)
0 2 * * * docker exec f1pa_postgres pg_dump -U f1pa f1pa_db > /backups/f1pa_$(date +\%Y\%m\%d).sql
```

#### Restore

```bash
# Restore depuis backup
docker exec -i f1pa_postgres psql -U f1pa -d f1pa_db < backup_20250129.sql
```

#### Backup volumes Docker

```bash
# Backup MLflow artifacts
tar -czf mlartifacts_backup.tar.gz mlartifacts/

# Backup Grafana data
docker run --rm -v f1pa_grafana_data:/data -v $(pwd):/backup alpine tar -czf /backup/grafana_backup.tar.gz -C /data .

# Restore
tar -xzf mlartifacts_backup.tar.gz
docker run --rm -v f1pa_grafana_data:/data -v $(pwd):/backup alpine tar -xzf /backup/grafana_backup.tar.gz -C /data
```

### Nettoyage cache Python

```bash
# Nettoyer __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
rm -rf htmlcov/ .coverage
```

---

## 📚 Ressources

- [README principal](README.md) - Vue d'ensemble du projet
- [Scripts README](scripts/README.md) - Documentation scripts utilitaires
- [Monitoring README](monitoring/README.md) - Guide monitoring détaillé
- [RGPD](RGPD.md) - Conformité RGPD
- [API Documentation](http://localhost:8000/docs) - Swagger UI


