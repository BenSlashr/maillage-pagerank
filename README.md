# SEO Internal Linking Tool - Maillage Pré-Publication

Outil de maillage interne SEO utilisant FastAPI pour le backend et HTML/CSS avec Alpine.js pour le frontend. Cette application permet d'analyser le contenu d'un site web, de générer des suggestions de maillage interne basées sur la similarité sémantique, et de visualiser l'impact de ces suggestions sur le PageRank des pages.

## Architecture

Cette application utilise une architecture simplifiée :

- **Backend** : FastAPI (Python)
- **Frontend** : HTML/CSS avec Alpine.js (sans React)
- **Communication en temps réel** : WebSockets
- **Analyse sémantique** : Modèles BERT via sentence-transformers
- **Crawling intégré** : Crawler Python basé sur requests et BeautifulSoup

## Fonctionnalités

### Fonctionnalités de base
- Upload de fichiers Excel (contenu, liens existants, GSC)
- Configuration des règles de maillage entre segments
- Analyse sémantique avec BERT
- Suivi de progression en temps réel
- Visualisation et téléchargement des résultats

### Nouvelles fonctionnalités
- **Crawler intégré** : Crawl automatique d'un site web avec filtrage des pages HTTP 200 uniquement
- **Gestion des redirections** : Suivi automatique des redirections et utilisation de l'URL finale
- **Déduplication des suggestions** : Élimination des suggestions en double pour une meilleure lisibilité
- **Compatibilité multi-formats** : Support des exports Screaming Frog et du crawler intégré
- **Normalisation des chemins de fichiers** : Gestion robuste des chemins relatifs/absolus
- **Normalisation des noms de colonnes** : Compatibilité entre différents formats de fichiers
- **Logs détaillés** : Journalisation améliorée pour faciliter le débogage

## Structure du projet

```
maillage-web-simple/
├── app/
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   ├── templates/
│   │   ├── index.html
│   │   ├── analysis.html
│   │   ├── rules.html
│   │   ├── results.html
│   │   └── pagerank_report.html
│   ├── uploads/
│   │   ├── content/
│   │   ├── links/
│   │   └── gsc/
│   ├── results/
│   ├── models/
│   │   ├── seo_analyzer.py       # Analyse sémantique et génération de suggestions
│   │   ├── crawler.py            # Crawler intégré pour l'analyse de sites
│   │   ├── pagerank.py           # Calcul du PageRank et visualisation
│   │   ├── pagerank_cache.py     # Cache pour les calculs de PageRank
│   │   ├── normalize_columns.py  # Normalisation des noms de colonnes
│   │   └── priority_manager.py   # Gestion des URLs prioritaires
│   └── main.py                   # Points d'entrée FastAPI et logique principale
└── requirements.txt
```

## Installation

1. Créez un environnement virtuel Python :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```

2. Installez les dépendances avec les versions spécifiques requises :
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note importante** : Les versions spécifiques suivantes sont requises pour assurer la compatibilité :
   - sentence-transformers==2.2.2
   - huggingface-hub==0.14.1 (version spécifique requise pour éviter les problèmes d'importation)
   - torch>=1.6.0
   - transformers>=4.6.0,<4.30.0

3. Téléchargez les ressources NLTK :
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```

## Lancement de l'application

1. Lancez l'application :
   ```bash
   python -m app.main
   ```
   Ou en se plaçant dans le dossier app :
   ```bash
   cd app && python main.py
   ```

2. Ouvrez votre navigateur et accédez à `http://localhost:8000`

3. Suivez les étapes dans l'interface :
   - **Option 1** : Uploadez vos fichiers Excel (contenu, liens, GSC)
   - **Option 2** : Utilisez le crawler intégré pour analyser un site web
   - Configurez les règles de maillage entre segments
   - Lancez l'analyse
   - Visualisez et téléchargez les résultats
   - Consultez le rapport de PageRank pour évaluer l'impact des suggestions

## Guide technique des fonctionnalités

### 1. Crawler intégré

Le crawler intégré permet d'analyser automatiquement un site web et de générer les fichiers nécessaires à l'analyse de maillage.

**Fonctionnement :**
- Implémenté dans `app/models/crawler.py`
- Utilise `requests` pour récupérer les pages et `BeautifulSoup` pour parser le HTML
- Suit automatiquement les redirections avec `allow_redirects=True`
- Filtre les pages pour ne conserver que celles avec un code HTTP 200
- Génère deux fichiers Excel :
  - **Fichier de contenu** : Contient les URLs, titres, H1, contenu extrait et statut HTTP
  - **Fichier de liens** : Contient les liens internes entre les pages valides (code 200)

**Colonnes générées :**
- Fichier de contenu : `Adresse`, `Titre`, `H1`, `Extracteur 1 1`, `Segments`, `Status`
- Fichier de liens : `Source`, `Destination`, `Interne`, `Texte`

### 2. Normalisation des noms de colonnes

Le système accepte désormais différents formats de noms de colonnes pour une meilleure compatibilité.

**Fonctionnement :**
- Implémenté dans `app/models/normalize_columns.py`
- Détecte automatiquement les colonnes source/destination dans différents formats :
  - Source : `Source`, `source`, `URL source`, etc.
  - Destination : `Destination`, `destination`, `target`, `Target`, etc.
- Normalise les noms de colonnes pour le calcul du PageRank et l'analyse

### 3. Déduplication des suggestions

Les suggestions de maillage sont maintenant dédupliquées pour éviter les doublons.

**Fonctionnement :**
- Implémenté dans `app/models/seo_analyzer.py` (méthode `_generate_suggestions`)
- Trie les suggestions par score de similarité décroissant
- Supprime les doublons en conservant la suggestion avec la meilleure similarité
- Ajoute des logs détaillés sur le nombre de suggestions avant/après déduplication

### 4. Normalisation des chemins de fichiers

La gestion des chemins de fichiers a été améliorée pour éviter les erreurs de fichiers introuvables.

**Fonctionnement :**
- Implémenté dans `app/main.py` (fonction `normalize_file_path`)
- Gère automatiquement les chemins relatifs et absolus
- Assure que tous les fichiers (contenu, liens, GSC) sont correctement résolus
- Utilisé dans les fonctions `analyze_content` et `run_analysis`

### 5. Filtrage des pages HTTP 200

Le système ne conserve désormais que les pages avec un code HTTP 200 pour le plan de maillage.

**Fonctionnement :**
- Implémenté dans `app/models/crawler.py`
- Filtre les pages lors du crawl et de la génération des fichiers
- Filtre également les liens pour ne conserver que ceux entre pages valides
- Ajoute une colonne `Status` dans le fichier de contenu pour faciliter le debug

## Différences avec la version React

Cette version utilise une approche plus simple et plus légère que la version originale avec React :

- Pas de build process complexe (pas de webpack, babel, etc.)
- Utilisation d'Alpine.js pour la réactivité côté client
- Templates HTML rendus côté serveur avec Jinja2
- Communication en temps réel via WebSockets
- Moins de dépendances JavaScript
- **Même backend FastAPI** avec les mêmes fonctionnalités

## Avantages de cette approche

- **Simplicité** : Pas besoin de gérer deux serveurs séparés
- **Performance** : Moins de JavaScript à charger
- **Maintenance** : Une seule base de code à maintenir
- **Déploiement** : Plus simple à déployer (un seul serveur)

## Débogage et maintenance

### Logs détaillés

L'application dispose désormais de logs détaillés pour faciliter le débogage :

- Statuts HTTP des pages crawlées
- Redirections suivies et URLs finales
- Mapping des colonnes détectées dans les fichiers
- Déduplication des suggestions
- Normalisation des chemins de fichiers

### Points d'attention pour les développeurs

1. **Noms de colonnes** : Le système normalise automatiquement les noms de colonnes, mais il est préférable d'utiliser des noms cohérents dans tout le code.

2. **Formats de fichiers** : L'application accepte différents formats de fichiers (Screaming Frog, crawler intégré), mais il est important de vérifier que les colonnes essentielles sont présentes.

3. **Chemins de fichiers** : Utilisez la fonction `normalize_file_path` pour garantir que les chemins sont corrects, surtout après un crawl.

4. **PageRank** : Le calcul du PageRank nécessite des colonnes `Source` et `Destination` (avec majuscules), mais le système normalise désormais automatiquement les noms de colonnes.

## Prochaines améliorations possibles

1. **Tests unitaires** : Ajouter des tests pour la robustesse du mapping de colonnes et la déduplication
2. **Documentation API** : Documenter les endpoints FastAPI pour faciliter l'intégration
3. **Interface de configuration du crawler** : Améliorer les options de configuration du crawler (profondeur, filtres, etc.)
4. **Visualisation améliorée** : Enrichir la visualisation du graphe de liens et du PageRank
5. **Export multi-formats** : Ajouter des options d'export pour différents formats (CSV, JSON, etc.)

## Format des fichiers d'entrée

### Fichier de contenu (obligatoire)
- Format : Excel (.xlsx)
- Colonnes requises : 
  - `Adresse` : URL de la page
  - `Segments` : Type de page (blog, catégorie, produit, etc.)
  - `Extracteur 1 1` : Contenu textuel de la page

### Fichier de liens existants (optionnel)
- Format : Excel (.xlsx)
- Colonnes requises :
  - `Source` : URL de la page source
  - `Destination` : URL de la page de destination

### Fichier GSC (optionnel)
- Format : Excel (.xlsx)
- Colonnes requises :
  - `URL` : URL de la page
  - `Clics` : Nombre de clics
  - `Impressions` : Nombre d'impressions
  - `Position` : Position moyenne
