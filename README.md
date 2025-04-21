# SEO Internal Linking Tool - Version Simplifiée

Une version simplifiée de l'outil de maillage interne SEO utilisant FastAPI pour le backend et HTML/CSS avec Alpine.js pour le frontend.

## Architecture

Cette application utilise une architecture simplifiée :

- **Backend** : FastAPI (Python)
- **Frontend** : HTML/CSS avec Alpine.js (sans React)
- **Communication en temps réel** : WebSockets

## Fonctionnalités

- Upload de fichiers Excel (contenu, liens existants, GSC)
- Configuration des règles de maillage entre segments
- Analyse sémantique avec BERT
- Suivi de progression en temps réel
- Visualisation et téléchargement des résultats

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
│   │   └── results.html
│   ├── uploads/
│   │   ├── content/
│   │   ├── links/
│   │   └── gsc/
│   ├── results/
│   ├── models/
│   │   └── seo_analyzer.py
│   └── main.py
└── requirements.txt
```

## Installation

1. Créez un environnement virtuel Python :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Téléchargez les ressources NLTK :
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```

## Lancement de l'application

1. Démarrez le serveur (assurez-vous que l'environnement virtuel est activé) :
   ```bash
   cd /Users/benoit/maillage-web-simple
   python app/main.py
   ```
   *Note : Lancer directement le script `app/main.py` évite des problèmes de compatibilité rencontrés avec `uvicorn --reload` sur certaines configurations.*

2. Accédez à l'application dans votre navigateur :
   ```
   http://localhost:8002
   ```

## Différences avec la version React

Cette version simplifiée utilise :

- **Alpine.js** au lieu de React pour la réactivité côté client
- **Templates Jinja2** au lieu de composants React
- **Bootstrap** pour le style au lieu de Material-UI
- **Même backend FastAPI** avec les mêmes fonctionnalités

## Avantages de cette approche

- **Simplicité** : Pas besoin de gérer deux serveurs séparés
- **Performance** : Moins de JavaScript à charger
- **Maintenance** : Une seule base de code à maintenir
- **Déploiement** : Plus simple à déployer (un seul serveur)

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
