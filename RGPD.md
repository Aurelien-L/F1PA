# Conformité RGPD - F1PA (Formula 1 Predictive Assistant)
---

## 📋 Résumé exécutif

Le projet F1PA traite **exclusivement des données publiques** issues de sources officielles (OpenF1 API, Wikipedia, Meteostat). **Aucune donnée personnelle identifiante (PII)** n'est collectée, stockée ou traitée.

Cette documentation présente la **démarche de conformité RGPD** du projet, même si le règlement ne s'applique pas strictement en l'absence de données personnelles.

---

## 1. Nature des données collectées

### 1.1 Sources de données

| Source | Type de données | Nature | Finalité |
|--------|----------------|--------|----------|
| **OpenF1 API** | Sessions F1, circuits, laps, pilotes | Publiques sportives | Agrégation dataset ML |
| **Wikipedia** | Coordonnées GPS circuits | Publiques référentielles | Enrichissement géographique |
| **Meteostat** | Données météorologiques | Publiques environnementales | Enrichissement contexte météo |

### 1.2 Données traitées

**Données sportives** :
- Identifiants sessions (meeting_key, session_key)
- Numéros de pilotes (driver_number) - **pas de noms dans le traitement ML**
- Performances : temps au tour, vitesses, secteurs
- Contexte : circuit, année, numéro de tour

**Données météorologiques** :
- Température, humidité, pression atmosphérique
- Vent (vitesse, direction)
- Horodatage UTC

**Données géographiques** :
- Coordonnées GPS circuits (latitude, longitude)
- Noms de villes, pays

### 1.3 Analyse RGPD

✅ **Aucune donnée personnelle identifiante** :
- Pas d'emails, adresses, numéros de téléphone
- Pas de données biométriques
- Pas de données de santé
- Pas de données sensibles (origine ethnique, opinions politiques, etc.)

⚠️ **Données publiques sur les pilotes** :
- Noms de pilotes présents dans `dim_drivers` (table de référence)
- Considérées comme **données publiques professionnelles** (sportifs de haut niveau)
- Non utilisées pour décisions automatisées à leur égard
- Finalité : contexte statistique uniquement

**Conclusion** : Le projet ne relève **pas du champ d'application strict du RGPD** (Article 2.2.c - données manifestement rendues publiques par la personne concernée).

---

## 2. Base légale et finalité du traitement

### 2.1 Base légale (hypothétique)

Si le RGPD s'appliquait, la base légale serait :
- **Article 6.1.f** : Intérêt légitime (recherche, éducation, innovation)
- **Article 6.1.e** : Mission d'intérêt public (recherche scientifique)

### 2.2 Finalité du traitement

**Finalité principale** :
- Développement d'un système de prédiction ML des performances en Formule 1
- Projet d'IA appliquée au sport automobile

**Finalités accessoires** :
- Analyse statistique des performances sportives
- Étude de l'impact des conditions météorologiques
- Démonstration de compétences Data Engineering / MLOps

**Principe de minimisation** : ✅
- Seules les données strictement nécessaires sont collectées
- Pas de collecte opportuniste ou excessive

---

## 3. Registre des traitements (Article 30 RGPD)

### Traitement #1 : Extraction de données (ETL - Extract)

| Élément | Description |
|---------|-------------|
| **Finalité** | Collecte de données brutes depuis APIs publiques |
| **Base légale** | Données publiques, intérêt légitime |
| **Catégories de données** | Sessions F1, circuits, météo, performances sportives |
| **Catégories de personnes** | Pilotes F1 (données publiques professionnelles) |
| **Destinataires** | Équipe projet F1PA uniquement |
| **Transferts hors UE** | OpenF1 API (peut être hébergée hors UE) - données publiques |
| **Durée conservation** | 2 ans (données brutes) |
| **Mesures sécurité** | Authentification API, logs traçabilité |

### Traitement #2 : Transformation et agrégation (ETL - Transform)

| Élément | Description |
|---------|-------------|
| **Finalité** | Nettoyage, enrichissement, construction dataset ML |
| **Base légale** | Intérêt légitime |
| **Catégories de données** | Dataset agrégé 71,645 laps avec features météo |
| **Catégories de personnes** | Pilotes F1 (anonymisés par driver_number dans ML) |
| **Destinataires** | Équipe projet, modèle ML |
| **Durée conservation** | 1 an après obsolescence du modèle |
| **Mesures sécurité** | Validation qualité données, logs transformation |

### Traitement #3 : Stockage en base de données (ETL - Load)

| Élément | Description |
|---------|-------------|
| **Finalité** | Stockage structuré pour requêtes et API |
| **Base légale** | Intérêt légitime |
| **Catégories de données** | Tables dim_circuits, dim_drivers, dim_sessions, fact_laps |
| **Destinataires** | API REST, applications consommatrices |
| **Durée conservation** | Durée de vie du projet (2-3 ans) |
| **Mesures sécurité** | Authentification PostgreSQL, containerisation Docker |

### Traitement #4 : Entraînement de modèles ML

| Élément | Description |
|---------|-------------|
| **Finalité** | Création modèle prédictif temps au tour |
| **Base légale** | Intérêt légitime |
| **Catégories de données** | Features numériques (vitesses, météo, performance) |
| **Pseudonymisation** | Driver_number utilisé (pas de noms en features) |
| **Durée conservation** | Modèles conservés tant que performants (1-2 ans) |
| **Mesures sécurité** | MLflow tracking, versioning modèles |

### Traitement #5 : API de prédictions

| Élément | Description |
|---------|-------------|
| **Finalité** | Exposition du modèle pour prédictions temps réel |
| **Base légale** | Intérêt légitime |
| **Catégories de données** | Features entrée (circuit, météo, driver_perf_score) |
| **Destinataires** | Application Streamlit, clients API autorisés |
| **Conservation logs** | 90 jours roulants (logs applicatifs) |
| **Mesures sécurité** | Authentification HTTP Basic, CORS, validation schémas |

---

## 4. Politique de rétention des données

### 4.1 Durées de conservation

| Catégorie | Durée | Justification |
|-----------|-------|---------------|
| **Données brutes Extract** | 2 ans | Reproductibilité pipeline, archive |
| **Données Transform** | 1 an | Réentraînement modèles |
| **Dataset ML final** | 1 an après obsolescence modèle | Traçabilité, audit |
| **Modèles ML** | Tant que performants (max 2 ans) | Utilisation production |
| **Logs API** | 90 jours | Debugging, monitoring |
| **Métriques MLflow** | 3 ans | Historique expérimentations |
| **Base PostgreSQL** | Durée projet (2-3 ans) | Accès données via API |

### 4.2 Procédure de purge

**Automatisation** :
- Script `etl/extract/run_extract_all.py --purge-raw` : suppression données brutes obsolètes
- Logs rotatifs : 90 jours automatique (configuration système)

**Manuelle** :
- Révision annuelle : suppression modèles dépréciés
- Archivage : export final pour documentation projet

---

## 5. Mesures de sécurité techniques et organisationnelles

### 5.1 Sécurité technique

**Contrôle d'accès** :
- ✅ Authentification API : HTTP Basic Auth (credentials `f1pa:f1pa`)
- ✅ Base données PostgreSQL : user/password dédiés
- ✅ Containerisation Docker : isolation services

**Intégrité des données** :
- ✅ Validation schémas Pydantic (API)
- ✅ Contraintes d'intégrité SQL (foreign keys, NOT NULL)
- ✅ Logs traçabilité : manifests JSON à chaque étape ETL

**Disponibilité** :
- ✅ Docker Compose : redémarrage automatique services
- ✅ Backups PostgreSQL possibles via volumes Docker
- ⚠️ Pas de haute disponibilité

### 5.2 Sécurité organisationnelle

**Formation** :
- Sensibilisation RGPD et bonnes pratiques
- Documentation des procédures

**Traçabilité** :
- Manifests JSON : horodatage, paramètres, versions
- Git : historique des modifications code
- MLflow : tracking complet expérimentations ML

**Limitations** :
- ⚠️ Credentials hardcodés dans code (amélioration : variables d'environnement)
- ⚠️ Pas de chiffrement données au repos (données publiques, risque faible)
- ⚠️ Pas d'audit logs centralisés (amélioration future : ELK stack)

---

## 6. Droits des personnes concernées

### 6.1 Analyse des droits RGPD

Bien que les données soient publiques, voici l'analyse des droits :

**Droit d'accès (Article 15)** :
- Non applicable : données publiques sportives
- Si demande : export des données pilote depuis API `/data/drivers?driver_number=X`

**Droit de rectification (Article 16)** :
- Non applicable : données issues de sources officielles (OpenF1)
- Responsabilité de la source primaire (FIA/Formula 1)

**Droit à l'effacement / "droit à l'oubli" (Article 17)** :
- Non applicable : données publiques, intérêt public (sport)
- Si demande exceptionnelle : suppression records dans dim_drivers, cascade sur fact_laps

**Droit à la limitation du traitement (Article 18)** :
- Non applicable dans le contexte actuel

**Droit à la portabilité (Article 20)** :
- Format machine-readable disponible : JSON (API), CSV (exports)
- Endpoint dédié possible : `GET /data/drivers/{id}/export`

**Droit d'opposition (Article 21)** :
- Non applicable : pas de marketing, pas de profilage à des fins décisionnelles

### 6.2 Procédure de demande

En cas de demande d'un pilote (hypothétique) :
1. **Contact** : email projet ou formulaire dédié
2. **Vérification identité** : preuve d'identité (protection usurpation)
3. **Traitement** : 1 mois maximum (Article 12.3)
4. **Réponse** : export données ou justification refus (données publiques)

**Contact RGPD** : *[À définir si projet en production]*

---

## 7. Analyse d'impact (PIA - Privacy Impact Assessment)

### 7.1 Évaluation des risques

| Risque | Probabilité | Gravité | Mesures d'atténuation |
|--------|-------------|---------|----------------------|
| **Fuite de données personnelles** | 🟢 Très faible | 🟢 Faible | Aucune PII collectée |
| **Accès non autorisé API** | 🟡 Moyenne | 🟡 Moyenne | Authentification HTTP Basic |
| **Perte de données** | 🟡 Moyenne | 🟡 Moyenne | Backups Docker volumes, reproductibilité pipeline |
| **Usurpation d'identité pilote** | 🟢 Très faible | 🟢 Faible | Données publiques, pas de décisions automatisées |
| **Profilage discriminatoire** | 🟢 Très faible | 🟢 Faible | Prédiction sportive uniquement, pas RH/assurance |

**Conclusion PIA** : ✅ Risque résiduel **FAIBLE**. Aucune mesure RGPD additionnelle requise.

### 7.2 Proportionnalité

**Test de proportionnalité** :
- ✅ Finalité légitime : développement système prédictif ML
- ✅ Nécessité : données strictement requises pour prédiction ML
- ✅ Proportionnalité : pas de collecte excessive
- ✅ Équilibre : intérêt légitime > droits personnes (données publiques)

---

## 8. Transferts de données hors Union Européenne

### 8.1 Identification des transferts

| Destinataire | Pays | Données | Garanties |
|--------------|------|---------|-----------|
| **OpenF1 API** | Probablement USA/CDN | Requêtes API (metadata) | Données publiques, HTTPS |
| **Meteostat** | Allemagne (UE) | Requêtes météo | Pas de transfert hors UE |
| **Wikipedia** | USA (Wikimedia Foundation) | Scraping pages publiques | Données publiques, robots.txt respecté |

### 8.2 Conformité Schrems II

**Analyse** :
- ✅ Données publiques : pas de restrictions RGPD
- ✅ Pas de données sensibles transférées
- ⚠️ Si évolution : implémenter clauses contractuelles types (CCT)

---

## 9. Sous-traitance et responsabilités

### 9.1 Services tiers

| Service | Rôle | Données traitées | Statut RGPD |
|---------|------|------------------|-------------|
| **OpenF1** | Fournisseur données | Données F1 publiques | Non sous-traitant (source publique) |
| **MLflow** | Tracking ML | Métriques, modèles | Hébergé en local (Docker) |
| **PostgreSQL** | Stockage | Dataset complet | Hébergé en local (Docker) |
| **Docker Hub** | Registry images | Pas de données projet | Infrastructure uniquement |

**Aucun sous-traitant RGPD** : tous les traitements sont réalisés localement.

### 9.2 Responsabilité

**Responsable du traitement** : Aurélien LEVA
**DPO (Data Protection Officer)** : Non requis (pas d'entreprise)

---

## 10. Documentation et traçabilité

### 10.1 Artefacts de conformité

**Fichiers de documentation** :
- ✅ Ce document : `RGPD.md`
- ✅ Schéma BD : `etl/load/schema.sql` (structure données)
- ✅ README projet : documentation architecture
- ✅ Manifests ETL : `data/extract/manifest_*.json` (traçabilité)

**Logs et audits** :
- Logs extraction : stdout scripts Python
- Logs API : console FastAPI / Uvicorn
- Tracking ML : MLflow UI (expérimentations)
- Git : historique commits

### 10.2 Révision de la conformité

**Fréquence** : Annuelle ou lors de modifications majeures

**Événements déclencheurs** :
- Ajout de nouvelles sources de données
- Changement de finalité (ex: commercialisation)
- Évolution réglementaire RGPD

---

## 11. Déclaration de conformité

### 11.1 Engagement

Le projet F1PA s'engage à :
- ✅ Traiter uniquement des données publiques
- ✅ Respecter les principes RGPD (minimisation, finalité, transparence)
- ✅ Maintenir des mesures de sécurité appropriées
- ✅ Documenter les traitements de données
- ✅ Répondre aux demandes de droits (si applicable)

### 11.2 Limitations du projet

En cas de passage en production commerciale, les mesures suivantes seraient requises :

- 🔴 Audit RGPD complet par un DPO
- 🔴 Clauses contractuelles avec fournisseurs données
- 🔴 Politique de confidentialité publique
- 🔴 Formulaires de consentement (si collecte étendue)
- 🔴 Registre des traitements formalisé (format CNIL)
- 🔴 Étude d'impact PIA approfondie (si données sensibles ajoutées)

---

## 12. Conclusion

### ✅ Synthèse de conformité

| Principe RGPD | Conformité | Commentaire |
|---------------|-----------|-------------|
| **Licéité** | ✅ Conforme | Données publiques, intérêt légitime |
| **Finalité** | ✅ Conforme | Finalité définie (ML prédictif) |
| **Minimisation** | ✅ Conforme | Uniquement données nécessaires |
| **Exactitude** | ✅ Conforme | Sources officielles (OpenF1, Meteostat) |
| **Conservation limitée** | ✅ Conforme | Politique rétention définie (2 ans max) |
| **Intégrité/Confidentialité** | ✅ Conforme | Authentification, containerisation |
| **Responsabilité** | ✅ Conforme | Documentation traçable |

### 🎯 Statut final

**Le projet F1PA est CONFORME aux exigences RGPD** dans son contexte actuel (données publiques).

Cette documentation présente la **démarche de conformité méthodologique** appliquée au projet, même si le règlement ne s'applique pas strictement en l'absence de données personnelles identifiantes.

---

## 📚 Références

**Réglementation** :
- [RGPD - Texte officiel (EUR-Lex)](https://eur-lex.europa.eu/eli/reg/2016/679/oj)
- [CNIL - Guide du développeur](https://www.cnil.fr/fr/guide-developpeur)
- [CNIL - Registre des traitements](https://www.cnil.fr/fr/RGDP-le-registre-des-activites-de-traitement)

**Bonnes pratiques** :
- [ISO 27001 - Sécurité de l'information](https://www.iso.org/standard/27001)
- [OWASP - Sécurité applications web](https://owasp.org/)
- [MLOps - Gouvernance des modèles ML](https://ml-ops.org/)

---

**Document établi le** : 26 janvier 2026  
**Version** : 1.0  
**Responsable** : Aurélien LEVA

