# F1PA - Streamlit Dashboard

Interface utilisateur pour le prédicteur de temps au tour F1PA.

Dashboard interactif permettant de prédire les temps au tour des pilotes F1 **avant qu'ils ne roulent**, basé sur leur performance historique et les conditions de course.

## Fonctionnalités

### Onglet Prediction
- **Sélection du pilote** avec photo officielle et couleur d'équipe
- **Sélection du circuit** avec informations pays et ville
- **Paramètres de course ajustables** :
  - Numéro de tour dans la session
  - Vitesses attendues (speed trap, intermédiaires 1 & 2)
  - Conditions météo (température, humidité, pression)
- **Prédiction en temps réel** avec temps formaté (mm:ss.xxx)
- **Comparaison intelligente** : différence vs moyennes circuit/pilote
- **Métriques du modèle** utilisé pour la prédiction

### Onglet Model
- **Informations détaillées** sur le modèle actif
- **Métriques de performance** :
  - MAE (Mean Absolute Error) de test et cross-validation
  - R² (coefficient de détermination)
  - RMSE et ratio d'overfitting
- **Détails MLflow** : run name, run ID, stratégie de sélection
- **Source du modèle** : MLflow (production) ou local (fallback)

### Onglet Links
- Liens directs vers **FastAPI Docs** (Swagger UI)
- Liens vers **MLflow UI** (tracking et artifacts)
- Lien vers le **repository GitHub**
- **Status des services** : API, PostgreSQL, MLflow (avec indicateurs visuels)

## Architecture

Le dashboard Streamlit fait partie d'une stack complète orchestrée par Docker Compose :

```
┌─────────────────┐
│   Streamlit     │  Port 8501 - Interface utilisateur
│   (Frontend)    │
└────────┬────────┘
         │ HTTP API calls
         ▼
┌─────────────────┐
│   FastAPI       │  Port 8000 - API REST + ML Service
│   (Backend)     │
└────┬───────┬────┘
     │       │
     │       └──────► MLflow (Port 5000) - Tracking & Artifacts
     │
     └──────────────► PostgreSQL (Port 5432) - Data Warehouse
```

## Déploiement

### Méthode recommandée : Docker Compose

Depuis la racine du projet :

```bash
# Démarrer tous les services (Postgres, MLflow, API, Streamlit)
docker-compose up -d

# Vérifier le status
docker-compose ps

# Voir les logs
docker logs f1pa_streamlit --tail 50
```

Le dashboard sera accessible sur **http://localhost:8501**

### Développement local

Si vous souhaitez développer l'interface sans Docker :

```bash
# Installer les dépendances
cd streamlit/
pip install -r requirements.txt

# Démarrer seulement l'infra (API, DB, MLflow)
cd ..
docker-compose up -d postgres mlflow api

# Lancer Streamlit en local
streamlit run streamlit/app.py
```

## Configuration

### Variables d'environnement

Configurées automatiquement dans Docker Compose (`docker-compose.yml`) :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `API_BASE_URL` | `http://api:8000` (Docker)<br>`http://localhost:8000` (local) | URL de l'API F1PA |
| `API_USERNAME` | `f1pa` | Username pour l'authentification API |
| `API_PASSWORD` | `f1pa` | Password pour l'authentification API |
| `MLFLOW_URL` | `http://localhost:5000` | URL de l'interface MLflow UI |

### Fichiers de configuration

```
streamlit/
├── app.py              # Application principale Streamlit
├── config.py           # Configuration et constantes
├── requirements.txt    # Dépendances Python (streamlit, requests, pandas)
├── Dockerfile          # Image Docker optimisée (~300 MB)
└── README.md           # Ce fichier
```

**Note** : Le `requirements.txt` local est **nécessaire** pour Docker. Il contient uniquement les dépendances frontend (pas de ML libs), ce qui garde l'image légère.

## Design

Interface inspirée de l'identité visuelle Formule 1 :

- **Noir** (`#15151E`) : Fond principal, cartes
- **Blanc** (`#FFFFFF`) : Texte, titres
- **Rouge F1** (`#E10600`) : Boutons, accents, call-to-actions
- **Gris** (`#38383F`, `#AAAAAA`) : Éléments secondaires, bordures

Photos officielles des pilotes et couleurs d'équipes via l'API Formula1.com.

## Guide d'utilisation

### Prédire un temps au tour

1. **Sélectionner un pilote** dans la liste déroulante
2. **Sélectionner un circuit** (ex: Monaco, Monza, Spa-Francorchamps)
3. **Ajuster les paramètres** (optionnel) :
   - Numéro de tour (1-70)
   - Vitesses attendues en km/h (basées sur historique)
   - Conditions météo (température, humidité, pression)
4. **Cliquer sur "🏎️ Predict Lap Time"**
5. **Analyser la prédiction** :
   - Temps au tour prédit (format mm:ss.xxx)
   - Différence vs moyenne circuit
   - Différence vs moyenne pilote
   - Métriques du modèle utilisé

### Consulter les informations du modèle

Onglet **Model** pour voir :
- Famille de modèle (Random Forest, XGBoost, etc.)
- Stratégie de sélection (MAE, robust)
- Métriques de test et cross-validation
- Run ID MLflow pour traçabilité

### Accéder aux services

Onglet **Links** pour :
- Tester l'API directement (Swagger UI)
- Voir les runs MLflow (experiments, artifacts)
- Vérifier le status des services (indicateurs de santé)

## Healthcheck

Le conteneur Streamlit inclut un healthcheck automatique :

```dockerfile
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
```

Vérifie que l'application Streamlit répond correctement toutes les 30 secondes.

**Note** : Si `curl` n'est pas installé dans l'image, le healthcheck échouera mais l'application fonctionnera normalement. Pour corriger, ajouter `curl` au Dockerfile :

```dockerfile
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
```

## Troubleshooting

### Streamlit ne démarre pas
```bash
# Vérifier les logs
docker logs f1pa_streamlit

# Redémarrer le service
docker-compose restart streamlit
```

### Erreur de connexion à l'API
- Vérifier que l'API est démarrée : `docker ps | grep f1pa_api`
- Tester l'API : `curl -u f1pa:f1pa http://localhost:8000/predict/model`
- Vérifier les variables d'environnement dans `docker-compose.yml`

### Photos de pilotes manquantes
- Vérifier que PostgreSQL contient les données : `docker exec f1pa_postgres psql -U f1pa -d f1pa_db -c "SELECT COUNT(*) FROM dim_drivers;"`
- Vérifier que l'API retourne `headshot_url` : `curl -u f1pa:f1pa http://localhost:8000/data/drivers | jq '.[0]'`

### Métriques du modèle indisponibles (N/A)
- Vérifier que MLflow est accessible : `curl http://localhost:5000/`
- Vérifier les logs API : `docker logs f1pa_api | grep MLflow`
- Si le modèle est en mode `local` (fallback), c'est normal que les métriques soient absentes
