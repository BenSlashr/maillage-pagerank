from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
import pandas as pd
import numpy as np
import os
import json
import logging
import uuid
import asyncio
import time
import shutil
import tempfile
import sys
import re
import traceback
import aiofiles
from typing import Dict, List, Optional, Any, Callable, Union, Set
from fastapi.concurrency import run_in_threadpool
import openpyxl
from fastapi.encoders import jsonable_encoder
from urllib.parse import urlparse

# Classe d'encodeur JSON personnalisée pour gérer les valeurs flottantes hors limites
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if np.isnan(obj):
                return None
            if np.isinf(obj):
                return 1.0e308 if obj > 0 else -1.0e308
            return round(float(obj), 6)  # Arrondir à 6 décimales
        return super().default(obj)

# Ajouter le répertoire parent au chemin pour les imports relatifs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import des modules
from models.pagerank import calculate_pagerank, calculate_pagerank_with_suggestions, prepare_graph_data, calculate_weighted_pagerank, calculate_weighted_pagerank_with_suggestions
from models.priority_urls import PriorityURLManager
from models.pagerank_cache import PageRankCache
from models.crawler import WebCrawler
from models.segment_rules import SegmentRuleManager
from models.seo_analyzer import SEOAnalyzer
from models.filter_self_links import filter_self_links

# Initialisation du cache PageRank (5 minutes par défaut)
pagerank_cache = PageRankCache(cache_duration_seconds=300)

# Fonction utilitaire pour gérer les valeurs flottantes hors limites (inf, -inf, NaN)
def safe_float(value):
    """Convertit une valeur flottante en une valeur compatible JSON"""
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if np.isinf(value):
                return 1.0e308 if value > 0 else -1.0e308
            if np.isnan(value):
                return None  # Utiliser None au lieu de 0.0 pour être cohérent avec l'encodeur JSON
            return round(float(value), 6)  # Arrondir à 6 décimales pour éviter des problèmes de précision
        return value
    except Exception:
        # En cas d'erreur, retourner None pour éviter les problèmes de sérialisation
        return None

# Fonction utilitaire pour sérialiser des objets en JSON en gérant les valeurs flottantes hors limites
def safe_json_dumps(obj):
    """Sérialise un objet en JSON en gérant les valeurs flottantes hors limites"""
    return json.dumps(obj, cls=CustomJSONEncoder)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('seo_analysis_api.log')
    ]
)

# Créons une classe de réponse JSON personnalisée plutôt que de modifier FastAPI
class CustomJSONResponse(JSONResponse):
    def render(self, content):
        # Pré-traitement des données pour remplacer les valeurs NaN, inf, -inf
        def clean_nan_values(obj):
            if isinstance(obj, dict):
                return {k: clean_nan_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan_values(item) for item in obj]
            elif isinstance(obj, float):
                if np.isnan(obj):
                    return None
                if np.isinf(obj):
                    return 1.0e308 if obj > 0 else -1.0e308
                return obj
            else:
                return obj
        
        # Nettoyer les données avant de les sérialiser
        cleaned_content = clean_nan_values(content)
        
        # Sérialiser avec notre encodeur JSON personnalisé
        return json.dumps(cleaned_content, cls=CustomJSONEncoder).encode("utf-8")

app = FastAPI(
    title="SEO Internal Linking API",
    description="API pour l'analyse et l'optimisation du maillage interne pour le SEO",
    version="1.0.0",
    # Utiliser notre réponse JSON personnalisée comme classe de réponse par défaut
    default_response_class=CustomJSONResponse
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Création des dossiers pour les uploads et les résultats
os.makedirs("app/uploads/content", exist_ok=True)
os.makedirs("app/uploads/links", exist_ok=True)
os.makedirs("app/uploads/gsc", exist_ok=True)
os.makedirs("app/results", exist_ok=True)

# Montage du dossier static pour servir les fichiers statiques
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Configuration des templates Jinja2
templates = Jinja2Templates(directory="app/templates")

# Dictionnaire pour suivre les tâches en cours
jobs = {}

# Gestionnaire de connexions WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        logging.info(f"Nouvelle connexion WebSocket pour la tâche {job_id}, total: {len(self.active_connections[job_id])}")

    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            if websocket in self.active_connections[job_id]:
                self.active_connections[job_id].remove(websocket)
                logging.info(f"Déconnexion WebSocket pour la tâche {job_id}, restant: {len(self.active_connections[job_id])}")

    async def send_job_update(self, job_id: str, data: dict):
        if job_id in self.active_connections and self.active_connections[job_id]:
            logging.info(f"Envoi de mise à jour WebSocket pour la tâche {job_id} à {len(self.active_connections[job_id])} clients")
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(data)
                except Exception as e:
                    logging.error(f"Erreur lors de l'envoi de la mise à jour WebSocket: {str(e)}")

manager = ConnectionManager()

# Variable globale pour l'instance de l'analyseur SEO
# Initialisée à None, sera créée lors de la première analyse
seo_analyzer = None

# Modèles Pydantic pour la validation des données
class LinkingRule(BaseModel):
    min_links: int
    max_links: int

class SegmentRules(BaseModel):
    rules: Dict[str, Dict[str, LinkingRule]]

class AnalysisConfig(BaseModel):
    min_similarity: float = 0.2
    anchor_suggestions: int = 3
    priority_urls: str = ""
    priority_urls_strict: bool = False

# Routes pour les pages HTML
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request):
    return templates.TemplateResponse("analysis.html", {"request": request})

@app.get("/rules", response_class=HTMLResponse)
async def rules_page(request: Request):
    return templates.TemplateResponse("rules_editor.html", {"request": request})

@app.get("/crawler")
async def get_crawler_page(request: Request):
    """Affiche la page du crawler"""
    return templates.TemplateResponse("crawler.html", {"request": request})

@app.get("/segment_rules")
async def get_segment_rules_page(request: Request):
    """Affiche la page de gestion des règles de segmentation"""
    return templates.TemplateResponse("segment_rules.html", {"request": request})

@app.get("/results/{job_id}", response_class=HTMLResponse)
async def read_results(request: Request, job_id: str):
    """Page de résultats de l'analyse"""
    return templates.TemplateResponse("results.html", {"request": request, "job_id": job_id})

@app.get("/visualization/{job_id}", response_class=HTMLResponse)
async def read_visualization(request: Request, job_id: str):
    """Page de visualisation du maillage interne"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")
        
    job = jobs[job_id]
    if job["status"] != "completed":
        # Rediriger vers la page de résultats si l'analyse n'est pas terminée
        return RedirectResponse(url=f"/results/{job_id}")
        
    return templates.TemplateResponse("visualization.html", {"request": request, "job_id": job_id})

@app.get("/cytoscape/{job_id}", response_class=HTMLResponse)
async def read_cytoscape_visualization(request: Request, job_id: str):
    """Page de visualisation du maillage interne avec Cytoscape"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")
        
    job = jobs[job_id]
    if job["status"] != "completed":
        # Rediriger vers la page de résultats si l'analyse n'est pas terminée
        return RedirectResponse(url=f"/results/{job_id}")
        
    return templates.TemplateResponse("cytoscape.html", {"request": request, "job_id": job_id})

@app.get("/pagerank-report/{job_id}", response_class=HTMLResponse)
async def read_pagerank_report(request: Request, job_id: str):
    """Page de rapport PageRank"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")
        
    job = jobs[job_id]
    if job["status"] != "completed":
        # Rediriger vers la page de résultats si l'analyse n'est pas terminée
        return RedirectResponse(url=f"/results/{job_id}")
        
    return templates.TemplateResponse("pagerank_report.html", {"request": request, "job_id": job_id})

# Routes API
@app.post("/api/upload/content")
async def upload_content_file(file: UploadFile = File(...)):
    logging.info(f"Requête reçue pour /api/upload/content: {file.filename}")
    try:
        file_path = await save_uploaded_file(file, "content")
        required_columns_content = ["Adresse", "Segments", "Extracteur 1 1"]
        validation_result = await run_in_threadpool(validate_excel_file, file_path, required_columns=required_columns_content)

        if not validation_result["valid"]:
            if os.path.exists(file_path):
                os.remove(file_path)
            logging.error(f"Fichier contenu invalide supprimé: {file_path} - Raison: {validation_result['message']}")
            raise HTTPException(status_code=400, detail=validation_result['message'])

        logging.info(f"Fichier contenu uploadé et validé: {file_path}")
        return {"filename": file.filename, "saved_path": file_path, "message": "Fichier contenu uploadé avec succès.", "valid": True}
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Erreur lors de l'upload du fichier contenu: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur lors de l'upload: {str(e)}")

@app.post("/api/upload/links")
async def upload_links_file(file: UploadFile = File(...)):
    logging.info(f"Requête reçue pour /api/upload/links: {file.filename}")
    try:
        file_path = await save_uploaded_file(file, "links")
        required_columns_links = ["Source", "Destination"]
        validation_result = await run_in_threadpool(validate_excel_file, file_path, required_columns=required_columns_links)

        if not validation_result["valid"]:
            if os.path.exists(file_path):
                os.remove(file_path)
            logging.error(f"Fichier liens invalide supprimé: {file_path} - Raison: {validation_result['message']}")
            raise HTTPException(status_code=400, detail=validation_result['message'])

        logging.info(f"Fichier liens uploadé et validé: {file_path}")
        return {"filename": file.filename, "saved_path": file_path, "message": "Fichier liens uploadé avec succès.", "valid": True}
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Erreur lors de l'upload du fichier liens: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur lors de l'upload: {str(e)}")

@app.post("/api/upload/gsc")
async def upload_gsc_file(file: UploadFile = File(...)):
    logging.info(f"Requête reçue pour /api/upload/gsc: {file.filename}")
    try:
        file_path = await save_uploaded_file(file, "gsc")
        required_columns_gsc = ["Query", "Page", "Clicks", "Impressions", "CTR", "Position"]
        validation_result = await run_in_threadpool(validate_excel_file, file_path, required_columns=required_columns_gsc)

        if not validation_result["valid"]:
            if os.path.exists(file_path):
                os.remove(file_path)
            logging.error(f"Fichier GSC invalide supprimé: {file_path} - Raison: {validation_result['message']}")
            raise HTTPException(status_code=400, detail=validation_result['message'])

        logging.info(f"Fichier GSC uploadé et validé: {file_path}")
        return {"filename": file.filename, "saved_path": file_path, "message": "Fichier GSC uploadé avec succès.", "valid": True}
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Erreur lors de l'upload du fichier GSC: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur lors de l'upload: {str(e)}")

@app.get("/api/segments")
async def get_segments(content_file: str):
    """Récupère les segments uniques du fichier de contenu"""
    try:
        if not os.path.exists(content_file):
            raise HTTPException(status_code=404, detail="Fichier de contenu non trouvé")

        df = pd.read_excel(content_file)
        if "Segments" not in df.columns:
            raise HTTPException(status_code=400, detail="Colonne 'Segments' non trouvée dans le fichier")

        segments = sorted(set(
            segment.lower().strip()
            for segment in df["Segments"].dropna().unique()
        ))

        # Normaliser les segments
        normalized_segments = sorted(set(
            'blog' if 'blog' in s or 'article' in s
            else 'categorie' if 'categ' in s
            else 'produit' if 'produit' in s or 'product' in s
            else s
            for s in segments
        ))

        return {"segments": normalized_segments}
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des segments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rules")
async def set_linking_rules(rules: SegmentRules):
    """Définit les règles de maillage entre segments"""
    try:
        # Sauvegarder les règles dans un fichier
        with open("app/segment_rules.json", "w", encoding="utf-8") as f:
            json.dump(rules.rules, f, ensure_ascii=False, indent=4)

        return {"message": "Règles de maillage enregistrées avec succès"}
    except Exception as e:
        logging.error(f"Erreur lors de l'enregistrement des règles de maillage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rules")
async def get_linking_rules():
    """Récupère les règles de maillage configurées"""
    try:
        if os.path.exists("app/segment_rules.json"):
            with open("app/segment_rules.json", "r", encoding="utf-8") as f:
                rules = json.load(f)
            return {"rules": rules}
        else:
            return {"rules": {}}
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des règles de maillage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rules")
def get_rules():
    try:
        rules_file = os.path.join("data", "rules", "segment_rules.json")
        if not os.path.exists(rules_file):
            logging.warning("Fichier de règles introuvable, retour d'un objet vide")
            return {"rules": {}}
        
        with open(rules_file, "r", encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                logging.warning("Fichier de règles vide, retour d'un objet vide")
                return {"rules": {}}
                
            rules = json.loads(content)
            
        # Convertir les valeurs de chaînes en nombres
        numeric_rules = {}
        for source in rules:
            numeric_rules[source] = {}
            for target in rules[source]:
                try:
                    numeric_rules[source][target] = int(float(rules[source][target]))
                except (ValueError, TypeError):
                    numeric_rules[source][target] = 0
        
        logging.info(f"Règles chargées avec succès: {len(numeric_rules)} segments")
        return {"rules": numeric_rules}
    except json.JSONDecodeError as e:
        logging.error(f"Erreur de décodage JSON lors de la récupération des règles: {str(e)}")
        return {"rules": {}}
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des règles: {str(e)}")
        return {"rules": {}}

@app.get("/api/rules/download")
def download_rules():
    try:
        rules_file = os.path.join("data", "rules", "segment_rules.json")
        if not os.path.exists(rules_file):
            raise HTTPException(status_code=404, detail="Fichier de règles non trouvé")
        
        return FileResponse(
            path=rules_file,
            filename="segment_rules.json",
            media_type="application/json"
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logging.error(f"Erreur lors du téléchargement des règles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class RulesData(BaseModel):
    rules: dict
    
    class Config:
        # Permettre des types arbitraires pour accepter les valeurs numériques dans les règles
        arbitrary_types_allowed = True

# Endpoint sans validation pour sauvegarder les règles
@app.post("/api/save-rules")
async def save_rules_no_validation(request: Request):
    try:
        # Lire le corps de la requête comme texte brut
        body = await request.body()
        body_str = body.decode('utf-8')
        
        # Parser manuellement le JSON
        rules_data = json.loads(body_str)
        
        if "rules" not in rules_data:
            logging.error("Format de règles invalide: clé 'rules' manquante")
            return {"success": False, "message": "Format de règles invalide"}
        
        # Créer le répertoire des règles s'il n'existe pas
        rules_dir = os.path.join("data", "rules")
        os.makedirs(rules_dir, exist_ok=True)
        
        # Convertir les valeurs en nombres entiers avant de sauvegarder
        numeric_rules = {}
        for source in rules_data["rules"]:
            numeric_rules[source] = {}
            for target in rules_data["rules"][source]:
                try:
                    # Convertir en nombre entier
                    value = rules_data["rules"][source][target]
                    if isinstance(value, str):
                        numeric_rules[source][target] = int(float(value))
                except (ValueError, TypeError):
                    numeric_rules[source][target] = 0
        
        # Sauvegarder les règles dans le fichier
        rules_file = os.path.join(rules_dir, "segment_rules.json")
        
        # Créer une sauvegarde du fichier existant si nécessaire
        if os.path.exists(rules_file):
            backup_file = os.path.join(rules_dir, f"segment_rules_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            shutil.copy2(rules_file, backup_file)
            logging.info(f"Sauvegarde créée: {backup_file}")
        
        # Écrire les nouvelles règles
        with open(rules_file, "w", encoding='utf-8') as f:
            json.dump(numeric_rules, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Règles sauvegardées avec succès: {len(numeric_rules)} segments")
        
        # Retourner les règles sauvegardées pour confirmation
        return {
            "success": True, 
            "message": "Règles sauvegardées avec succès",
            "rules": numeric_rules
        }
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde des règles: {str(e)}")
        return {"success": False, "message": str(e)}

@app.post("/api/save_content_rules")
async def save_content_rules(rules: str = Form(...)):
    """Sauvegarde les règles de segmentation pour l'analyse de contenu"""
    try:
        # Convertir les règles en objet JSON
        rules_data = json.loads(rules)
        
        # Convertir les règles en format numérique
        numeric_rules = []
        for rule in rules_data:
            numeric_rule = {
                "min_words": int(rule["min_words"]),
                "max_words": int(rule["max_words"]),
                "min_score": float(rule["min_score"]),
                "max_suggestions": int(rule["max_suggestions"]),
                "segment": rule["segment"]
            }
            numeric_rules.append(numeric_rule)
        
        # Sauvegarder les règles dans un fichier JSON
        with open("app/segment_rules.json", "w", encoding="utf-8") as f:
            json.dump(numeric_rules, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "message": "Règles sauvegardées avec succès",
            "rules": numeric_rules
        }
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde des règles: {str(e)}")
        return {"success": False, "message": str(e)}

@app.get("/api/segment_rules")
async def get_segment_rules():
    """Récupère les règles de segmentation"""
    try:
        # Initialiser le gestionnaire de règles
        rule_manager = SegmentRuleManager("app/data/segment_rules.json")
        
        # Exporter les règles
        rules_data = rule_manager.export_rules()
        
        return {
            "success": True,
            "rules": rules_data["rules"],
            "default_segment": rules_data["default_segment"]
        }
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des règles: {str(e)}")
        return {"success": False, "message": str(e)}

@app.post("/api/segment_rules")
async def update_segment_rules(data: Dict):
    """Met à jour les règles de segmentation"""
    try:
        # Initialiser le gestionnaire de règles
        rule_manager = SegmentRuleManager("app/data/segment_rules.json")
        
        # Importer les règles
        rule_manager.import_rules(data)
        
        # Sauvegarder les règles
        rule_manager.save_rules()
        
        return {
            "success": True,
            "message": "Règles de segmentation mises à jour avec succès"
        }
    except Exception as e:
        logging.error(f"Erreur lors de la mise à jour des règles: {str(e)}")
        return {"success": False, "message": str(e)}

def normalize_file_path(file_path: str) -> str:
    """Normalise un chemin de fichier pour s'assurer qu'il est correct"""
    if not file_path:
        return file_path
        
    # Si le chemin est déjà absolu ou commence par app/uploads, le retourner tel quel
    if os.path.isabs(file_path) or file_path.startswith('app/uploads/'):
        return file_path
        
    # Si le chemin commence par content/ ou links/ ou gsc/, ajouter app/uploads/
    if file_path.startswith('content/') or file_path.startswith('links/') or file_path.startswith('gsc/'):
        return os.path.join('app/uploads', file_path)
        
    # Sinon, considérer le chemin comme relatif au répertoire uploads
    return os.path.join('app/uploads', file_path)

@app.post("/api/analyze")
async def analyze_content(
    background_tasks: BackgroundTasks,
    content_file: str = Form(...),
    links_file: Optional[str] = Form(None),
    gsc_file: Optional[str] = Form(None),
    config: str = Form(...)
):
    """Lance l'analyse du contenu et génère des suggestions de maillage interne"""
    try:
        # Normaliser les chemins de fichiers
        content_file = normalize_file_path(content_file)
        if links_file:
            links_file = normalize_file_path(links_file)
        if gsc_file:
            gsc_file = normalize_file_path(gsc_file)
            
        # Journaliser les chemins de fichiers pour le débogage
        logging.info(f"Fichier de contenu normalisé: {content_file}")
        if links_file:
            logging.info(f"Fichier de liens normalisé: {links_file}")
        if gsc_file:
            logging.info(f"Fichier GSC normalisé: {gsc_file}")
        
        # Vérifier que les fichiers existent
        if not os.path.exists(content_file):
            logging.error(f"Fichier de contenu non trouvé: {content_file}")
            raise HTTPException(status_code=404, detail="Fichier de contenu non trouvé")

        if links_file and not os.path.exists(links_file):
            logging.error(f"Fichier de liens non trouvé: {links_file}")
            raise HTTPException(status_code=404, detail="Fichier de liens non trouvé")

        if gsc_file and not os.path.exists(gsc_file):
            logging.error(f"Fichier GSC non trouvé: {gsc_file}")
            raise HTTPException(status_code=404, detail="Fichier GSC non trouvé")

        # Charger la configuration
        try:
            config_data = json.loads(config)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Format de configuration invalide")

        # Créer un ID unique pour la tâche
        job_id = str(uuid.uuid4())

        # Initialiser les informations de la tâche
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Analyse en attente de démarrage",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "result_file": None,
            "content_file": content_file,
            "links_file": links_file,
            "gsc_file": gsc_file,
            "rules_file": "app/segment_rules.json" if os.path.exists("app/segment_rules.json") else None,
            "config": config_data
        }

        # Lancer l'analyse en arrière-plan
        background_tasks.add_task(
            run_analysis,
            job_id
        )

        return {"job_id": job_id}
    except Exception as e:
        # Journalisation détaillée de l'erreur
        logging.error(f"Erreur lors du lancement de l'analyse: {str(e)}")
        logging.error(f"Traceback complet: {traceback.format_exc()}")
        
        # Journalisation des états des fichiers
        logging.error(f"Fichier de contenu: {content_file}, existe: {os.path.exists(content_file) if content_file else False}")
        logging.error(f"Fichier de liens: {links_file}, existe: {os.path.exists(links_file) if links_file else False}")
        logging.error(f"Fichier GSC: {gsc_file}, existe: {os.path.exists(gsc_file) if gsc_file else False}")
        
        # Journalisation de la configuration
        try:
            logging.error(f"Configuration: {config}")
            logging.error(f"Configuration parsée: {json.loads(config)}")
        except Exception as config_err:
            logging.error(f"Erreur lors de l'analyse de la configuration: {str(config_err)}")
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/job/{job_id}")
async def check_job_status(job_id: str):
    """Vérifie le statut d'une tâche"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")

    return jobs[job_id]

@app.get("/api/results/{job_id}")
async def get_results(job_id: str, format: str = "json"):
    """Récupère les résultats d'une analyse terminée"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")

    job = jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="L'analyse n'est pas encore terminée")

    result_file_path = job.get("result_file")
    if not result_file_path or not os.path.exists(result_file_path):
        raise HTTPException(status_code=404, detail="Fichier de résultats non trouvé")

    if format.lower() == "json":
        try:
            df = pd.read_excel(result_file_path)
            
            # Convertir le DataFrame en dictionnaire
            df_dict = df.to_dict(orient="records")
            
            # Utiliser notre classe de réponse JSON personnalisée
            return CustomJSONResponse(
                content={"results": df_dict}
            )
        except Exception as e:
            logging.error(f"Erreur lors de la lecture du fichier de résultats: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture du fichier de résultats: {str(e)}")
    else:
        # Renvoyer le fichier Excel
        return FileResponse(
            path=result_file_path,
            filename=f"resultats_maillage_{job_id}.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Endpoint pour la visualisation du graphe
@app.get("/api/graph/{job_id}")
async def get_graph_data(job_id: str):
    """Récupère les données de graphe pour la visualisation du maillage interne"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")

    job = jobs[job_id]
    
    # Vérifier que les fichiers nécessaires existent
    content_file = job.get("content_file")
    links_file = job.get("links_file")
    result_file = job.get("result_file")
    
    if not content_file or not os.path.exists(content_file):
        raise HTTPException(status_code=404, detail="Fichier de contenu non trouvé")
        
    try:
        # Charger les données de contenu
        content_df = pd.read_excel(content_file)
        
        # Charger les liens existants si disponibles
        existing_links_df = pd.DataFrame(columns=["Source", "Destination"])
        if links_file and os.path.exists(links_file):
            existing_links_df = pd.read_excel(links_file)
            
        # Vérifier si le fichier de liens est vide ou ne contient pas de données valides
        if existing_links_df.empty:
            logging.warning(f"Le fichier de liens est vide ou ne contient pas de données valides pour le job {job_id}")
            return {
                "nodes": [],
                "edges": {"current": [], "suggested": [], "combined": []},
                "metrics": {"node_count": 0, "current_edge_count": 0, "suggested_edge_count": 0, "improvement_percentage": 0},
                "error": "Le fichier de liens est vide ou ne contient pas de données valides. Assurez-vous d'avoir fourni un fichier de liens avec des colonnes 'Source' et 'Destination'."
            }
        
        # Charger les suggestions de liens si disponibles
        suggested_links_df = None
        if result_file and os.path.exists(result_file) and job["status"] == "completed":
            suggested_links_df = pd.read_excel(result_file)
        
        # Calculer le PageRank
        current_pagerank, optimized_pagerank = None, None
        if not existing_links_df.empty:
            if suggested_links_df is not None:
                # Convertir le DataFrame de suggestions au format attendu
                suggested_links_for_pagerank = pd.DataFrame({
                    "Source": suggested_links_df["source_url"],
                    "Destination": suggested_links_df["target_url"]
                })
                current_pagerank, optimized_pagerank = calculate_pagerank_with_suggestions(
                    existing_links_df, suggested_links_for_pagerank
                )
            else:
                current_pagerank = calculate_pagerank(existing_links_df)
        
        # Préparer les données de graphe
        graph_data = prepare_graph_data(
            content_df, 
            existing_links_df, 
            suggested_links_df,
            current_pagerank,
            optimized_pagerank
        )
        
        # Vérifier si le graphe contient des données
        if not graph_data.get("nodes") or len(graph_data.get("nodes", [])) == 0:
            logging.warning(f"Aucune donnée de graphe générée pour le job {job_id} après filtrage des pages HTML")
            graph_data["error"] = "Aucune page HTML trouvée dans les données. Vérifiez que vos URLs correspondent à des pages HTML et non à des fichiers statiques."
        
        return graph_data
        
    except Exception as e:
        logging.error(f"Erreur lors de la préparation des données de graphe: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la préparation des données de graphe: {str(e)}")

# Endpoint pour la visualisation Cytoscape
@app.get("/api/visualization-data/{job_id}")
async def get_visualization_data(job_id: str, content_links_only: bool = False, use_weighted_pagerank: bool = False, alpha: float = 0.5, beta: float = 0.5):
    """Récupère les données combinées (graphe et PageRank) pour la visualisation Cytoscape"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")

    try:
        # Récupérer les données du graphe
        graph_data = await get_graph_data(job_id)
        
        # Récupérer les données de PageRank (standard ou pondéré)
        if use_weighted_pagerank:
            pagerank_data = await get_weighted_pagerank(job_id, content_links_only=content_links_only, alpha=alpha, beta=beta)
        else:
            pagerank_data = await get_pagerank(job_id, content_links_only=content_links_only)
        
        # Combiner les données
        visualization_data = {
            "graph": graph_data,
            "pagerank": pagerank_data,
            "config": {
                "use_weighted_pagerank": use_weighted_pagerank,
                "content_links_only": content_links_only,
                "alpha": alpha,
                "beta": beta
            }
        }
        
        return visualization_data
        
    except Exception as e:
        logging.error(f"Erreur lors de la préparation des données de visualisation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la préparation des données de visualisation: {str(e)}")

# Endpoints pour la gestion des URL prioritaires
@app.post("/api/priority-urls/{job_id}")
def add_priority_urls(job_id: str, urls: List[str]):
    """
    Ajoute des URL à la liste des URL prioritaires.
    
    Args:
        job_id: Identifiant de la tâche d'analyse
        urls: Liste des URL à ajouter
    """
    try:
        # Vérifier que la tâche existe
        job_file = os.path.join("app", "data", "jobs", job_id, "job.json")
        if not os.path.exists(job_file):
            raise HTTPException(status_code=404, detail="Tâche non trouvée")
        
        # Initialiser le gestionnaire d'URL prioritaires
        priority_manager = PriorityURLManager(job_id)
        
        # Ajouter les URL
        priority_manager.add_priority_urls(urls)
        
        return {"status": "success", "message": f"{len(urls)} URL ajoutées à la liste des URL prioritaires"}
    
    except Exception as e:
        logging.error(f"Erreur lors de l'ajout d'URL prioritaires: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/priority-urls/{job_id}")
def remove_priority_urls(job_id: str, urls: List[str]):
    """
    Supprime des URL de la liste des URL prioritaires.
    
    Args:
        job_id: Identifiant de la tâche d'analyse
        urls: Liste des URL à supprimer
    """
    try:
        # Vérifier que la tâche existe
        job_file = os.path.join("app", "data", "jobs", job_id, "job.json")
        if not os.path.exists(job_file):
            raise HTTPException(status_code=404, detail="Tâche non trouvée")
        
        # Initialiser le gestionnaire d'URL prioritaires
        priority_manager = PriorityURLManager(job_id)
        
        # Supprimer les URL
        priority_manager.remove_priority_urls(urls)
        
        return {"status": "success", "message": f"{len(urls)} URL supprimées de la liste des URL prioritaires"}
    
    except Exception as e:
        logging.error(f"Erreur lors de la suppression d'URL prioritaires: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/priority-urls/{job_id}")
def get_priority_urls(job_id: str):
    """
    Récupère la liste des URL prioritaires.
    
    Args:
        job_id: Identifiant de la tâche d'analyse
    """
    try:
        # Vérifier que le répertoire de la tâche existe
        job_dir = os.path.join("app", "data", "jobs", job_id)
        if not os.path.exists(job_dir):
            # Créer le répertoire s'il n'existe pas
            os.makedirs(job_dir, exist_ok=True)
            logging.info(f"Répertoire créé pour la tâche {job_id}")
        
        # Initialiser le gestionnaire d'URL prioritaires
        priority_manager = PriorityURLManager(job_id)
        
        # Récupérer les URL
        urls = priority_manager.get_priority_urls()
        
        return {"status": "success", "urls": urls}
    
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des URL prioritaires: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e), "urls": []}

@app.get("/api/priority-urls/{job_id}/impact")
def analyze_priority_urls_impact(job_id: str, content_links_only: bool = False, alpha: float = 0.5, beta: float = 0.5):
    """
    Analyse l'impact du plan de maillage sur les URL prioritaires.
    
    Args:
        job_id: Identifiant de la tâche d'analyse
        content_links_only: Si True, ne prend en compte que les liens dans le contenu principal
        alpha: Coefficient pour la pondération sémantique (entre 0 et 1)
        beta: Coefficient pour la pondération par position (entre 0 et 1)
    """
    try:
        # Vérifier que le répertoire de la tâche existe
        job_dir = os.path.join("app", "data", "jobs", job_id)
        if not os.path.exists(job_dir):
            # Créer le répertoire s'il n'existe pas
            os.makedirs(job_dir, exist_ok=True)
            logging.info(f"Répertoire créé pour la tâche {job_id}")
        
        # Initialiser le gestionnaire d'URL prioritaires
        priority_manager = PriorityURLManager(job_id)
        
        # Récupérer les URL prioritaires
        priority_urls = priority_manager.get_priority_urls()
        
        # Si aucune URL prioritaire n'est définie, retourner une réponse vide
        if not priority_urls:
            return {
                "status": "success", 
                "message": "Aucune URL prioritaire définie",
                "impact": {},
                "summary": {
                    "total": 0,
                    "improved": 0,
                    "maintained": 0,
                    "degraded": 0,
                    "not_found": 0
                }
            }
        
        try:
            # Récupérer les données de PageRank
            pagerank_data = get_weighted_pagerank(job_id, content_links_only, alpha, beta)
            
            # Analyser l'impact sur les URL prioritaires
            if "current" in pagerank_data and "optimized" in pagerank_data:
                current_pagerank = {url: data["pagerank"] for url, data in pagerank_data["current"].items()}
                optimized_pagerank = {url: data["pagerank"] for url, data in pagerank_data["optimized"].items()}
                
                impact_analysis = priority_manager.analyze_pagerank_impact(current_pagerank, optimized_pagerank)
                
                return {
                    "status": "success", 
                    "impact": impact_analysis,
                    "summary": {
                        "total": len(impact_analysis),
                        "improved": sum(1 for data in impact_analysis.values() if data["status"] == "improved"),
                        "maintained": sum(1 for data in impact_analysis.values() if data["status"] == "maintained"),
                        "degraded": sum(1 for data in impact_analysis.values() if data["status"] == "degraded"),
                        "not_found": sum(1 for data in impact_analysis.values() if data["status"] == "not_found")
                    }
                }
            else:
                return {
                    "status": "error", 
                    "message": "Données de PageRank non disponibles",
                    "impact": {},
                    "summary": {
                        "total": 0,
                        "improved": 0,
                        "maintained": 0,
                        "degraded": 0,
                        "not_found": 0
                    }
                }
        except Exception as inner_e:
            logging.error(f"Erreur lors de la récupération des données de PageRank: {str(inner_e)}", exc_info=True)
            return {
                "status": "error", 
                "message": f"Erreur lors de la récupération des données de PageRank: {str(inner_e)}",
                "impact": {},
                "summary": {
                    "total": len(priority_urls),
                    "improved": 0,
                    "maintained": 0,
                    "degraded": 0,
                    "not_found": len(priority_urls)
                }
            }
    
    except Exception as e:
        logging.error(f"Erreur lors de l'analyse de l'impact sur les URL prioritaires: {str(e)}", exc_info=True)
        return {
            "status": "error", 
            "message": str(e),
            "impact": {},
            "summary": {
                "total": 0,
                "improved": 0,
                "maintained": 0,
                "degraded": 0,
                "not_found": 0
            }
        }

# Endpoint pour le calcul du PageRank pondéré
@app.get("/api/weighted-pagerank/{job_id}")
def get_weighted_pagerank(job_id: str, content_links_only: bool = False, alpha: float = 0.5, beta: float = 0.5):
    """Calcule et retourne les scores PageRank pondérés pour les pages du site"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")

    job = jobs[job_id]
    
    # Vérifier que les fichiers nécessaires existent
    content_file = job.get("content_file")
    links_file = job.get("links_file")
    result_file = job.get("result_file")
    
    if not links_file or not os.path.exists(links_file):
        raise HTTPException(status_code=404, detail="Fichier de liens non trouvé")
    
    if not content_file or not os.path.exists(content_file):
        raise HTTPException(status_code=404, detail="Fichier de contenu non trouvé")
    
    # Vérifier si les résultats sont déjà en cache
    cached_results = pagerank_cache.get(job_id, content_links_only, alpha, beta)
    if cached_results:
        logging.info(f"Utilisation des résultats PageRank pondéré en cache pour job_id={job_id}, content_links_only={content_links_only}, alpha={alpha}, beta={beta}")
        return cached_results
        
    try:
        # Charger les données de contenu
        content_df = pd.read_excel(content_file)
        
        # Charger les liens existants
        existing_links_df = pd.read_excel(links_file)
        
        # Charger les suggestions de liens si disponibles
        suggested_links_df = None
        if result_file and os.path.exists(result_file) and job["status"] == "completed":
            suggested_links_df = pd.read_excel(result_file)
            # Convertir au format attendu
            suggested_links_df = pd.DataFrame({
                "Source": suggested_links_df["source_url"],
                "Destination": suggested_links_df["target_url"]
            })
        
        # Récupérer les URL prioritaires
        priority_urls = []
        priority_urls_strict = False
        
        try:
            # Initialiser le gestionnaire d'URL prioritaires
            priority_manager = PriorityURLManager(job_id)
            priority_urls = priority_manager.get_priority_urls()
            
            if priority_urls:
                logging.info(f"Prise en compte de {len(priority_urls)} URL prioritaires pour le calcul du PageRank")
                # Vérifier si le mode strict est activé
                config_file = os.path.join("app", "data", "jobs", job_id, "config.json")
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            priority_urls_strict = config.get("priorityUrlsStrict", False)
                    except Exception as e:
                        logging.error(f"Erreur lors de la lecture du fichier de configuration: {str(e)}")
                        
                logging.info(f"Mode strict pour les URL prioritaires: {priority_urls_strict}")
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des URL prioritaires: {str(e)}")
            # Continuer sans URL prioritaires
            priority_urls = []
        
        # Calculer le PageRank pondéré actuel et optimisé
        if suggested_links_df is not None:
            current_pagerank, optimized_pagerank = calculate_weighted_pagerank_with_suggestions(
                existing_links_df, 
                suggested_links_df,
                content_df=content_df,
                content_links_only=content_links_only,
                alpha=alpha,
                beta=beta,
                priority_urls=priority_urls,
                priority_urls_strict=priority_urls_strict
            )
            
            # Préparer les résultats en utilisant la fonction safe_float pour gérer les valeurs hors limites
            
            results = {
                "current": {
                    url: {"pagerank": safe_float(score), "rank": i+1} 
                    for i, (url, score) in enumerate(sorted(current_pagerank.items(), key=lambda x: x[1], reverse=True))
                },
                "optimized": {
                    url: {"pagerank": safe_float(score), "rank": i+1} 
                    for i, (url, score) in enumerate(sorted(optimized_pagerank.items(), key=lambda x: x[1], reverse=True))
                },
                "improvement": {},
                "config": {
                    "alpha": alpha,
                    "beta": beta,
                    "content_links_only": content_links_only
                }
            }
            
            # Calculer les améliorations
            for url in current_pagerank:
                if url in optimized_pagerank:
                    current_score = current_pagerank[url]
                    optimized_score = optimized_pagerank[url]
                    improvement_pct = ((optimized_score - current_score) / current_score) * 100 if current_score > 0 else 0
                    
                    results["improvement"][url] = {
                        "absolute": optimized_score - current_score,
                        "percentage": round(improvement_pct, 2),
                        "current_rank": results["current"][url]["rank"],
                        "optimized_rank": results["optimized"][url]["rank"],
                        "rank_change": results["current"][url]["rank"] - results["optimized"][url]["rank"]
                    }
            
            # Mettre en cache les résultats
            pagerank_cache.set(job_id, content_links_only, alpha, beta, results)
            logging.info(f"Résultats PageRank pondéré mis en cache pour job_id={job_id}, content_links_only={content_links_only}, alpha={alpha}, beta={beta}")
            
            return results
        else:
            # Seulement le PageRank pondéré actuel
            current_pagerank = calculate_weighted_pagerank(
                existing_links_df,
                content_df=content_df,
                content_links_only=content_links_only,
                alpha=alpha,
                beta=beta
            )
            
            # Préparer les résultats en utilisant la fonction safe_float pour gérer les valeurs hors limites
            
            results = {
                "current": {
                    url: {"pagerank": safe_float(score), "rank": i+1} 
                    for i, (url, score) in enumerate(sorted(current_pagerank.items(), key=lambda x: x[1], reverse=True))
                },
                "config": {
                    "alpha": alpha,
                    "beta": beta,
                    "content_links_only": content_links_only
                }
            }
            
            return results
    
    except Exception as e:
        logging.error(f"Erreur lors du calcul du PageRank pondéré: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul du PageRank pondéré: {str(e)}")

# Endpoint pour le calcul du PageRank
@app.get("/api/pagerank/{job_id}")
async def get_pagerank(job_id: str, content_links_only: bool = False):
    """Calcule et retourne les scores PageRank pour les pages du site"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")

    job = jobs[job_id]
    
    # Vérifier que les fichiers nécessaires existent
    links_file = job.get("links_file")
    result_file = job.get("result_file")
    
    if not links_file or not os.path.exists(links_file):
        raise HTTPException(status_code=404, detail="Fichier de liens non trouvé")
    
    # Paramètres pour le cache (valeurs par défaut pour le PageRank standard)
    alpha = 0.0
    beta = 0.0
    
    # Vérifier si les résultats sont déjà en cache
    cached_results = pagerank_cache.get(job_id, content_links_only, alpha, beta)
    if cached_results:
        logging.info(f"Utilisation des résultats PageRank en cache pour job_id={job_id}, content_links_only={content_links_only}")
        return cached_results
        
    try:
        # Charger les liens existants
        existing_links_df = pd.read_excel(links_file)
        
        # Charger les suggestions de liens si disponibles
        suggested_links_df = None
        if result_file and os.path.exists(result_file) and job["status"] == "completed":
            suggested_links_df = pd.read_excel(result_file)
            # Convertir au format attendu
            suggested_links_df = pd.DataFrame({
                "Source": suggested_links_df["source_url"],
                "Destination": suggested_links_df["target_url"]
            })
        
        # Calculer le PageRank actuel et optimisé
        if suggested_links_df is not None:
            current_pagerank, optimized_pagerank = calculate_pagerank_with_suggestions(
                existing_links_df, suggested_links_df,
                content_links_only=content_links_only
            )
            
            # Préparer les résultats
            results = {
                "current": {
                    url: {"pagerank": score, "rank": i+1} 
                    for i, (url, score) in enumerate(sorted(current_pagerank.items(), key=lambda x: x[1], reverse=True))
                },
                "optimized": {
                    url: {"pagerank": score, "rank": i+1} 
                    for i, (url, score) in enumerate(sorted(optimized_pagerank.items(), key=lambda x: x[1], reverse=True))
                },
                "improvement": {}
            }
            
            # Calculer les améliorations
            for url in current_pagerank:
                if url in optimized_pagerank:
                    current_score = current_pagerank[url]
                    optimized_score = optimized_pagerank[url]
                    improvement_pct = ((optimized_score - current_score) / current_score) * 100 if current_score > 0 else 0
                    
                    results["improvement"][url] = {
                        "absolute": optimized_score - current_score,
                        "percentage": round(improvement_pct, 2),
                        "current_rank": results["current"][url]["rank"],
                        "optimized_rank": results["optimized"][url]["rank"],
                        "rank_change": results["current"][url]["rank"] - results["optimized"][url]["rank"]
                    }
            
            # Mettre en cache les résultats
            pagerank_cache.set(job_id, content_links_only, alpha, beta, results)
            logging.info(f"Résultats PageRank mis en cache pour job_id={job_id}, content_links_only={content_links_only}")
            
            return results
        else:
            # Seulement le PageRank actuel
            current_pagerank = calculate_pagerank(existing_links_df, content_links_only=content_links_only)
            
            # Préparer les résultats
            results = {
                "current": {
                    url: {"pagerank": score, "rank": i+1} 
                    for i, (url, score) in enumerate(sorted(current_pagerank.items(), key=lambda x: x[1], reverse=True))
                }
            }
            
            # Mettre en cache les résultats
            pagerank_cache.set(job_id, content_links_only, alpha, beta, results)
            logging.info(f"Résultats PageRank (sans suggestions) mis en cache pour job_id={job_id}, content_links_only={content_links_only}")
            
            return results
            
    except Exception as e:
        logging.error(f"Erreur lors du calcul du PageRank: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul du PageRank: {str(e)}")

@app.get("/api/force-complete/{job_id}")
async def force_complete_job(job_id: str):
    """Force l'arrêt de l'analyse et renvoie les résultats"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")

    job = jobs[job_id]

    # Vérifier si l'analyse est déjà terminée
    if job["status"] == "completed":
        return job

    # Vérifier si l'analyse est en cours
    if job["status"] != "running":
        raise HTTPException(status_code=400, detail="L'analyse n'est pas en cours")

    # Forcer la fin de l'analyse
    job["status"] = "completed"
    job["progress"] = 100
    job["message"] = "Analyse terminée (forcée)"
    job["end_time"] = datetime.now().isoformat()

    # Vérifier s'il y a un fichier de résultats temporaire
    temp_result_file = f"app/results/temp_{job_id}.xlsx"
    if os.path.exists(temp_result_file):
        final_result_file = f"app/results/maillage_{job_id}.xlsx"
        shutil.copy(temp_result_file, final_result_file)
        job["result_file"] = final_result_file

    # Notifier les clients connectés
    await manager.send_job_update(job_id, job)

    return job

@app.get("/api/stop/{job_id}")
async def stop_job(job_id: str):
    """Arrête une analyse en cours"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")

    job = jobs[job_id]

    # Vérifier si l'analyse est déjà terminée
    if job["status"] in ["completed", "failed"]:
        return job

    # Marquer l'analyse comme arrêtée
    job["status"] = "failed"
    job["message"] = "Analyse arrêtée par l'utilisateur"
    job["end_time"] = datetime.now().isoformat()

    # Notifier les clients connectés
    await manager.send_job_update(job_id, job)

    return job

@app.get("/api/samples/{file_type}")
async def download_sample(file_type: str):
    """Télécharge un fichier exemple"""
    sample_files = {
        "content": "app/static/samples/exemple_contenu.xlsx",
        "links": "app/static/samples/exemple_liens.xlsx",
        "gsc": "app/static/samples/exemple_gsc.xlsx"
    }

    if file_type not in sample_files:
        raise HTTPException(status_code=400, detail="Type de fichier exemple non valide")

    file_path = sample_files[file_type]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier exemple non trouvé")

    return FileResponse(path=file_path)

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """Endpoint WebSocket pour les mises à jour en temps réel"""
    await manager.connect(websocket, job_id)
    try:
        # Envoyer l'état actuel de la tâche
        if job_id in jobs:
            await websocket.send_json(jobs[job_id])

        # Boucle de réception des messages
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)

# Endpoints pour le crawler web
@app.post("/api/start_crawl")
async def start_crawl(
    background_tasks: BackgroundTasks,
    start_url: str = Form(...),
    max_pages: int = Form(10000),
    respect_robots: bool = Form(True),
    crawl_delay: float = Form(0.5),
    exclude_patterns: str = Form("")
):
    """
    Démarre une tâche de crawling d'un site web.
    
    Args:
        start_url: URL de départ pour le crawl
        max_pages: Nombre maximum de pages à crawler
        respect_robots: Si True, respecte les règles du robots.txt
        crawl_delay: Délai entre les requêtes en secondes
    
    Returns:
        ID de la tâche de crawl
    """
    try:
        # Valider l'URL
        parsed_url = urlparse(start_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("URL invalide. Assurez-vous d'inclure le protocole (http:// ou https://)")
        
        # Générer un ID de tâche unique
        job_id = str(uuid.uuid4())
        
        # Initialiser la tâche dans le dictionnaire des tâches
        jobs[job_id] = {
            "id": job_id,
            "type": "crawl",
            "status": "initializing",
            "start_time": datetime.now().isoformat(),
            "progress": {
                "current": 0,
                "total": max_pages,
                "description": "Initialisation du crawl..."
            },
            "params": {
                "start_url": start_url,
                "max_pages": max_pages,
                "respect_robots": respect_robots,
                "crawl_delay": crawl_delay,
                "segment_rules_file": "app/data/segment_rules.json",
                "exclude_patterns": [pattern.strip() for pattern in exclude_patterns.split("\n") if pattern.strip()]
            },
            "results": {
                "content_file": None,
                "links_file": None
            },
            "stop_requested": False
        }
        
        logging.info(f"Nouvelle tâche de crawl créée: {job_id} pour {start_url}")
        
        # Lancer la tâche de crawl en arrière-plan
        background_tasks.add_task(run_crawl, job_id)
        
        return {"job_id": job_id, "status": "initializing"}
    
    except ValueError as e:
        logging.error(f"Erreur de validation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Erreur lors du démarrage du crawl: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.get("/api/crawl_status/{job_id}")
async def check_crawl_status(job_id: str):
    """
    Vérifie le statut d'une tâche de crawl.
    
    Args:
        job_id: ID de la tâche de crawl
    
    Returns:
        Statut et progression de la tâche
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Tâche {job_id} non trouvée")
    
    job = jobs[job_id]
    
    if job["type"] != "crawl":
        raise HTTPException(status_code=400, detail=f"La tâche {job_id} n'est pas une tâche de crawl")
    
    return {
        "id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "results": job["results"] if job["status"] == "completed" else None
    }

@app.get("/api/download_content_file/{job_id}")
async def download_content_file(job_id: str):
    """
    Télécharge le fichier de contenu généré par le crawler.
    
    Args:
        job_id: ID du job de crawl
        
    Returns:
        Fichier Excel de contenu
    """
    try:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} non trouvé")
            
        job = jobs[job_id]
        if job["type"] != "crawl" or job["status"] != "completed":
            raise HTTPException(status_code=400, detail="Ce job n'est pas un crawl terminé")
            
        content_file_path = job["results"].get("content_file")
        if not content_file_path or not os.path.exists(content_file_path):
            raise HTTPException(status_code=404, detail="Fichier de contenu non trouvé")
            
        # Extraire le nom du fichier à partir du chemin
        filename = os.path.basename(content_file_path)
        
        # Renvoyer le fichier avec un nom personnalisé
        domain = urlparse(job["params"]["start_url"]).netloc
        download_filename = f"contenu_{domain}_{datetime.now().strftime('%Y%m%d')}.xlsx"
        
        return FileResponse(
            path=content_file_path,
            filename=download_filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        logging.error(f"Erreur lors du téléchargement du fichier de contenu: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.get("/api/download_links_file/{job_id}")
async def download_links_file(job_id: str):
    """
    Télécharge le fichier de liens généré par le crawler.
    
    Args:
        job_id: ID du job de crawl
        
    Returns:
        Fichier Excel de liens
    """
    try:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} non trouvé")
            
        job = jobs[job_id]
        if job["type"] != "crawl" or job["status"] != "completed":
            raise HTTPException(status_code=400, detail="Ce job n'est pas un crawl terminé")
            
        links_file_path = job["results"].get("links_file")
        if not links_file_path or not os.path.exists(links_file_path):
            raise HTTPException(status_code=404, detail="Fichier de liens non trouvé")
            
        # Extraire le nom du fichier à partir du chemin
        filename = os.path.basename(links_file_path)
        
        # Renvoyer le fichier avec un nom personnalisé
        domain = urlparse(job["params"]["start_url"]).netloc
        download_filename = f"liens_{domain}_{datetime.now().strftime('%Y%m%d')}.xlsx"
        
        return FileResponse(
            path=links_file_path,
            filename=download_filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        logging.error(f"Erreur lors du téléchargement du fichier de liens: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.post("/api/test_exclude_pattern")
async def test_exclude_pattern(test_url: str = Form(...), pattern: str = Form(...)):
    """
    Teste si une URL correspond à un pattern d'exclusion.
    
    Args:
        test_url: URL à tester
        pattern: Pattern d'exclusion à tester
        
    Returns:
        Résultat du test (match ou non)
    """
    try:
        # Vérifier si le pattern est une regex
        is_regex = pattern.startswith("regex:")
        
        if is_regex:
            regex_pattern = pattern[6:]
            try:
                match = bool(re.search(regex_pattern, test_url))
                return {
                    "match": match,
                    "message": f"L'URL {'correspond' if match else 'ne correspond pas'} au pattern regex"
                }
            except re.error as e:
                return {
                    "match": False,
                    "error": True,
                    "message": f"Expression régulière invalide: {str(e)}"
                }
        else:
            # Test simple de présence du pattern dans l'URL
            match = pattern in test_url
            return {
                "match": match,
                "message": f"L'URL {'contient' if match else 'ne contient pas'} le pattern"
            }
    except Exception as e:
        logging.error(f"Erreur lors du test de pattern: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du test: {str(e)}")

@app.post("/api/stop_crawl/{job_id}")
async def stop_crawl(job_id: str):
    """
    Arrête une tâche de crawl en cours.
    
    Args:
        job_id: ID de la tâche de crawl
    
    Returns:
        Confirmation de l'arrêt
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Tâche {job_id} non trouvée")
    
    job = jobs[job_id]
    
    if job["type"] != "crawl":
        raise HTTPException(status_code=400, detail=f"La tâche {job_id} n'est pas une tâche de crawl")
    
    if job["status"] in ["completed", "failed"]:
        return {"success": False, "error": f"La tâche {job_id} est déjà terminée"}
    
    # Marquer la tâche comme devant être arrêtée
    job["stop_requested"] = True
    logging.info(f"Demande d'arrêt pour la tâche de crawl {job_id}")
    
    return {"success": True, "message": f"Demande d'arrêt envoyée pour la tâche {job_id}"}

# Fonction pour exécuter l'analyse en arrière-plan
async def run_analysis(job_id: str):
    """Exécute l'analyse SEO en arrière-plan"""
    global seo_analyzer
    job = jobs[job_id]

    # Mise à jour initiale du statut et notification
    job["status"] = "running"
    job["message"] = "Démarrage de l'analyse..."
    job["start_time"] = datetime.now().isoformat()
    await manager.send_job_update(job_id, job)

    # Récupération des informations nécessaires du job
    content_file = job.get("content_file")
    links_file = job.get("links_file")
    gsc_file = job.get("gsc_file")
    rules_file = job.get("rules_file")
    config = job.get("config", {})
    
    # Normaliser les chemins de fichiers
    content_file = normalize_file_path(content_file)
    if links_file:
        links_file = normalize_file_path(links_file)
    if gsc_file:
        gsc_file = normalize_file_path(gsc_file)
    if rules_file:
        rules_file = normalize_file_path(rules_file)
        
    # Mettre à jour les chemins normalisés dans le job
    job["content_file"] = content_file
    job["links_file"] = links_file
    job["gsc_file"] = gsc_file
    job["rules_file"] = rules_file

    # Log pour indiquer quels fichiers optionnels sont utilisés
    logging.info(f"[Job {job_id}] Fichier de contenu: {content_file} (existe: {os.path.exists(content_file) if content_file else False})")
    logging.info(f"[Job {job_id}] Fichier de liens existants: {links_file} (existe: {os.path.exists(links_file) if links_file else False})")
    logging.info(f"[Job {job_id}] Fichier GSC: {gsc_file} (existe: {os.path.exists(gsc_file) if gsc_file else False})")
    logging.info(f"[Job {job_id}] Fichier de règles: {rules_file} (existe: {os.path.exists(rules_file) if rules_file else False})")
    logging.info(f"[Job {job_id}] Configuration utilisée: {config}")

    try:
        # Journalisation détaillée pour le débogage
        logging.info(f"[Job {job_id}] Début de l'analyse avec les paramètres suivants:")
        logging.info(f"[Job {job_id}] - content_file: {content_file}")
        logging.info(f"[Job {job_id}] - links_file: {links_file}")
        logging.info(f"[Job {job_id}] - gsc_file: {gsc_file}")
        logging.info(f"[Job {job_id}] - rules_file: {rules_file}")
        logging.info(f"[Job {job_id}] - config: {config}")
        
        # Initialiser l'analyseur SEO s'il n'existe pas encore
        global seo_analyzer
        if seo_analyzer is None:
            logging.info(f"[Job {job_id}] Initialisation du modèle SEOAnalyzer (peut prendre du temps la première fois)...")
            from app.models.seo_analyzer import SEOAnalyzer
            
            # Créer une fonction d'adaptation pour le callback de progression
            # SEOAnalyzer appelle avec (description, current, total)
            # update_job_progress attend (job_id, current, total, description)
            async def progress_adapter(description: str, current, total):
                # Convertir explicitement en entiers pour éviter les erreurs de type
                try:
                    # S'assurer que current et total sont des valeurs numériques
                    # Si ce sont des chaînes, essayer de les nettoyer avant conversion
                    if isinstance(current, str):
                        # Supprimer tout caractère non numérique
                        current = ''.join(c for c in current if c.isdigit() or c == '.')
                    if isinstance(total, str):
                        # Supprimer tout caractère non numérique
                        total = ''.join(c for c in total if c.isdigit() or c == '.')
                        
                    current_int = int(float(current)) if current else 0
                    total_int = int(float(total)) if total else 1
                    
                    # Vérifier que les valeurs sont cohérentes
                    if current_int > total_int:
                        current_int = total_int
                        
                    await update_job_progress(job_id, current_int, total_int, description)
                except Exception as e:
                    logging.error(f"Erreur dans progress_adapter: {e} - description={description}, current={current}, total={total}")
            
            seo_analyzer = SEOAnalyzer(progress_callback=progress_adapter)
            logging.info(f"[Job {job_id}] Modèle SEOAnalyzer initialisé.")

        # Charger les règles de maillage si un fichier est spécifié
        linking_rules = None
        if rules_file and os.path.exists(rules_file):
            logging.info(f"[Job {job_id}] Chargement des règles depuis: {rules_file}")
            try:
                with open(rules_file, "r", encoding="utf-8") as f:
                    linking_rules = json.load(f)
                logging.info(f"[Job {job_id}] Règles chargées avec succès.")
            except Exception as rule_err:
                logging.warning(f"[Job {job_id}] Erreur lors du chargement des règles depuis {rules_file}: {rule_err}. Utilisation des règles par défaut.")

        # Traiter les URL prioritaires
        priority_urls = []
        priority_urls_strict = False
        
        if "priorityUrls" in config and config["priorityUrls"]:
            # Séparer les URL par ligne et nettoyer les espaces
            priority_urls = [url.strip() for url in config["priorityUrls"].split('\n') if url.strip()]
            priority_urls_strict = config.get("priorityUrlsStrict", False)
            
            if priority_urls:
                logging.info(f"[Job {job_id}] {len(priority_urls)} URL prioritaires définies")
                logging.info(f"[Job {job_id}] Mode strict pour les URL prioritaires: {priority_urls_strict}")
                
                # Créer le gestionnaire d'URL prioritaires et sauvegarder les URL
                priority_manager = PriorityURLManager(job_id)
                priority_manager.add_priority_urls(priority_urls)
        
        # Exécuter l'analyse
        logging.info(f"[Job {job_id}] Lancement de seo_analyzer.analyze...")
        result_file = await seo_analyzer.analyze(
            content_file=content_file,
            links_file=links_file,
            gsc_file=gsc_file,
            min_similarity=float(config.get("minSimilarity", 0.2)),  # Extraire et convertir en float
            anchor_suggestions=int(config.get("anchorSuggestions", 3)),  # Extraire et convertir en int
            linking_rules=linking_rules,
            priority_urls=priority_urls,
            priority_urls_strict=priority_urls_strict
        )
        logging.info(f"[Job {job_id}] Analyse terminée par seo_analyzer.analyze. Fichier résultat: {result_file}")
        
        # Filtrer les auto-liens dans le fichier de résultats
        try:
            logging.info(f"[Job {job_id}] Filtrage des auto-liens dans le fichier de résultats...")
            # Charger le fichier Excel
            suggestions_df = pd.read_excel(result_file, sheet_name="Suggestions")
            
            # Appliquer le filtre pour supprimer les auto-liens
            filtered_df = filter_self_links(suggestions_df)
            
            # Sauvegarder le fichier filtré
            with pd.ExcelWriter(result_file, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, sheet_name='Suggestions', index=False)
            
            logging.info(f"[Job {job_id}] Filtrage des auto-liens terminé. {len(suggestions_df) - len(filtered_df)} auto-liens supprimés.")
        except Exception as filter_err:
            logging.error(f"[Job {job_id}] Erreur lors du filtrage des auto-liens: {str(filter_err)}", exc_info=True)
            # Ne pas interrompre le processus en cas d'erreur lors du filtrage

        # Mettre à jour le statut de la tâche vers "completed"
        job["status"] = "completed"
        job["progress"] = 100
        job["message"] = "Analyse terminée avec succès."
        job["end_time"] = datetime.now().isoformat()
        job["result_file"] = result_file

        # Notifier les clients connectés de la fin
        await manager.send_job_update(job_id, job)

    except Exception as e:
        # Journalisation détaillée de l'erreur
        logging.error(f"[Job {job_id}] Erreur lors de l'analyse: {str(e)}", exc_info=True)
        logging.error(f"[Job {job_id}] Traceback complet: {traceback.format_exc()}")
        
        # Journalisation des états des fichiers
        if content_file and os.path.exists(content_file):
            logging.error(f"[Job {job_id}] Fichier de contenu existe: {content_file}, taille: {os.path.getsize(content_file)} octets")
        else:
            logging.error(f"[Job {job_id}] Fichier de contenu n'existe pas ou est invalide: {content_file}")
            
        if links_file and os.path.exists(links_file):
            logging.error(f"[Job {job_id}] Fichier de liens existe: {links_file}, taille: {os.path.getsize(links_file)} octets")
        else:
            logging.error(f"[Job {job_id}] Fichier de liens n'existe pas ou est invalide: {links_file}")
            
        if gsc_file and os.path.exists(gsc_file):
            logging.error(f"[Job {job_id}] Fichier GSC existe: {gsc_file}, taille: {os.path.getsize(gsc_file)} octets")
        else:
            logging.error(f"[Job {job_id}] Fichier GSC n'existe pas ou est invalide: {gsc_file}")

        # Mettre à jour le statut de la tâche en cas d'erreur
        job["status"] = "failed"
        job["message"] = f"Erreur: {str(e)}"
        job["end_time"] = datetime.now().isoformat()

        # Notifier les clients connectés de l'échec
        await manager.send_job_update(job_id, job)

# Fonction pour mettre à jour la progression d'une tâche
async def update_job_progress(job_id: str, description: str, current: int, total: int):
    """Met à jour la progression d'une tâche et notifie les clients"""
    if job_id not in jobs:
        return

    job = jobs[job_id]
    # Calculer le pourcentage de progression
    progress = int((current / max(1, total)) * 100)

    # Mettre à jour les informations de la tâche
    job["progress"] = progress
    job["message"] = description

    # Logguer la progression par intervalles réguliers
    if current > 0 and current % 10 == 0:
        logging.info(f"[Job {job_id}] Progression: {progress}% - {description}")

    # Notifier les clients connectés
    await manager.send_job_update(job_id, job)

# Fonction pour sauvegarder un fichier uploadé
async def save_uploaded_file(file: UploadFile, folder: str) -> str:
    """Sauvegarde un fichier uploadé dans le dossier spécifié et retourne le chemin complet."""
    start_time = datetime.now()
    logging.info(f"Début sauvegarde: {file.filename} dans {folder}")

    folder_path = f"app/uploads/{folder}"
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file.filename)

    logging.info(f"Chemin de sauvegarde: {file_path}")

    bytes_written = 0
    chunk_count = 0
    log_interval_chunks = 10

    try:
        async with aiofiles.open(file_path, "wb") as buffer:
            logging.info(f"Début lecture/écriture async boucle pour {file.filename}")
            while True:
                read_start = datetime.now()
                content = await file.read(1024 * 1024)
                read_end = datetime.now()

                if not content:
                    logging.info(f"Fin de lecture (contenu vide) pour {file.filename}")
                    break

                write_start = datetime.now()
                await buffer.write(content)
                write_end = datetime.now()

                bytes_written += len(content)
                chunk_count += 1

                if chunk_count % log_interval_chunks == 0:
                    logging.info(f"Chunk {chunk_count} pour {file.filename}: Read {(read_end - read_start).total_seconds():.4f}s, Write {(write_end - write_start).total_seconds():.4f}s. Total écrit: {bytes_written / (1024*1024):.2f} MB")

            logging.info(f"Fin lecture/écriture async boucle pour {file.filename}. Total Chunks: {chunk_count}, Total écrit: {bytes_written / (1024*1024):.2f} MB")

    finally:
        close_start = datetime.now()
        logging.info(f"Fermeture fichier upload: {file.filename}")
        await file.close()
        close_end = datetime.now()
        logging.info(f"Fichier upload fermé: {file.filename}. Durée fermeture: {(close_end - close_start).total_seconds():.4f}s")

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    logging.info(f"Fichier sauvegardé : {file_path}. Durée totale sauvegarde: {total_duration:.2f}s")
    return file_path

# Fonction pour valider un fichier Excel (version optimisée lisant uniquement les en-têtes)
def validate_excel_file(file_path: str, required_columns: List[str]) -> dict:
    """Valide qu'un fichier Excel contient les colonnes requises en lisant seulement les en-têtes."""
    try:
        # Ouvrir le classeur en mode lecture seule et data_only pour obtenir les valeurs calculées
        workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        sheet = workbook.active # Prendre la première feuille active
        
        # Lire la première ligne pour obtenir les en-têtes
        header_row = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), None)
        
        if not header_row:
            return {"valid": False, "message": "Le fichier Excel est vide ou la première ligne (en-têtes) n'a pas pu être lue."}
        
        # Nettoyer les noms d'en-têtes (supprimer espaces superflus, convertir en string au cas où)
        headers = [str(cell).strip() if cell is not None else "" for cell in header_row]
        
        missing_columns = []
        for column in required_columns:
            if column not in headers:
                missing_columns.append(column)
                
        if missing_columns:
            col_str = ", ".join([f"'{col}'" for col in missing_columns])
            message = f"La colonne {'s' if len(missing_columns) > 1 else ''} {col_str} {'sont requises mais non trouvées' if len(missing_columns) > 1 else 'est requise mais non trouvée'} dans le fichier."
            return {"valid": False, "message": message}
        
        return {"valid": True, "message": "Les colonnes requises sont présentes."}
        
    except FileNotFoundError:
        logging.error(f"validate_excel_file: Fichier non trouvé - {file_path}")
        return {"valid": False, "message": f"Erreur interne: Le fichier '{os.path.basename(file_path)}' n'a pas été trouvé pour validation."}
    except Exception as e:
        # Capturer d'autres erreurs potentielles (fichier corrompu, etc.)
        logging.error(f"Erreur lors de la validation Excel (en-têtes seulement) pour {file_path}: {e}", exc_info=True)
        return {"valid": False, "message": f"Impossible de valider le fichier Excel '{os.path.basename(file_path)}'. Il est peut-être corrompu ou n'est pas un fichier Excel valide. Erreur: {str(e)}"}
    finally:
        # Assurer la fermeture du classeur même si read_only=True
        if 'workbook' in locals() and workbook:
            try:
                workbook.close()
            except Exception as close_err:
                logging.warning(f"Erreur lors de la fermeture du classeur openpyxl pour {file_path}: {close_err}")


# Fonction pour exécuter le crawling en arrière-plan
async def run_crawl(job_id: str):
    """Exécute le crawling d'un site web en arrière-plan"""
    job = jobs[job_id]
    
    # Mise à jour initiale du statut et notification
    job["status"] = "running"
    await manager.send_job_update(job_id, {
        "type": "progress",
        "description": "Démarrage du crawl...",
        "current": 0,
        "total": job["params"]["max_pages"]
    })
    
    try:
        # Récupération des paramètres du job
        start_url = job["params"]["start_url"]
        max_pages = job["params"]["max_pages"]
        respect_robots = job["params"]["respect_robots"]
        crawl_delay = job["params"]["crawl_delay"]
        
        # Création d'une fonction de callback pour suivre la progression
        async def progress_callback(description, current, total):
            job["progress"] = {
                "description": description,
                "current": current,
                "total": total
            }
            # Vérifier si l'arrêt a été demandé
            if job["stop_requested"]:
                raise asyncio.CancelledError("Crawl arrêté par l'utilisateur")
            # Envoyer la mise à jour aux clients WebSocket
            await manager.send_job_update(job_id, {
                "type": "progress",
                "description": description,
                "current": current,
                "total": total
            })
        
        # Initialisation du crawler
        segment_rules_file = job["params"].get("segment_rules_file", "app/data/segment_rules.json")
        exclude_patterns = job["params"].get("exclude_patterns", [])
        
        # Journaliser les patterns d'exclusion
        if exclude_patterns:
            logging.info(f"Patterns d'URL exclus: {exclude_patterns}")
            await manager.send_job_update(job_id, {
                "type": "log",
                "message": f"Patterns d'URL exclus: {', '.join(exclude_patterns)}",
                "level": "info"
            })
        
        crawler = WebCrawler(
            start_url=start_url,
            max_pages=max_pages,
            respect_robots=respect_robots,
            crawl_delay=crawl_delay,
            segment_rules_file=segment_rules_file,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )
        
        # Envoi d'un message de log au client
        await manager.send_job_update(job_id, {
            "type": "log",
            "message": f"Démarrage du crawl pour {start_url} (max {max_pages} pages)",
            "level": "info"
        })
        
        # Exécution du crawl
        content_file_path, links_file_path = await crawler.crawl()
        
        # Mise à jour des résultats du job
        job["results"] = {
            "content_file": content_file_path,
            "links_file": links_file_path
        }
        
        # Récupération des statistiques du crawl
        stats = crawler.get_stats()
        
        # Envoi des statistiques au client
        await manager.send_job_update(job_id, {
            "type": "stats",
            "stats": stats
        })
        
        # Mise à jour du statut du job
        job["status"] = "completed"
        job["end_time"] = datetime.now().isoformat()
        
        # Notification de fin aux clients WebSocket
        await manager.send_job_update(job_id, {
            "type": "complete",
            "content_file": os.path.basename(content_file_path),
            "links_file": os.path.basename(links_file_path)
        })
        
        logging.info(f"Crawl terminé pour la tâche {job_id}")
        logging.info(f"Fichiers générés: {content_file_path}, {links_file_path}")
        
    except asyncio.CancelledError as e:
        # Le crawl a été arrêté par l'utilisateur
        job["status"] = "stopped"
        job["end_time"] = datetime.now().isoformat()
        job["error"] = str(e)
        
        await manager.send_job_update(job_id, {
            "type": "log",
            "message": f"Crawl arrêté: {str(e)}",
            "level": "warning"
        })
        
        logging.warning(f"Crawl arrêté pour la tâche {job_id}: {str(e)}")
        
    except Exception as e:
        # Une erreur s'est produite pendant le crawl
        job["status"] = "failed"
        job["end_time"] = datetime.now().isoformat()
        job["error"] = str(e)
        
        await manager.send_job_update(job_id, {
            "type": "error",
            "message": f"Erreur pendant le crawl: {str(e)}"
        })
        
        logging.error(f"Erreur pendant le crawl pour la tâche {job_id}: {str(e)}", exc_info=True)

if __name__ == "__main__":
    import uvicorn

    # Création des répertoires nécessaires au démarrage
    try:
        os.makedirs("app/uploads/content", exist_ok=True)
        os.makedirs("app/uploads/links", exist_ok=True)
        os.makedirs("app/uploads/gsc", exist_ok=True)
        os.makedirs("app/results", exist_ok=True)
        os.makedirs("app/static/samples", exist_ok=True)
        logging.info("Répertoires nécessaires vérifiés/créés.")
    except Exception as e:
        logging.error(f"Erreur lors de la création des répertoires: {e}")

    logging.info("Lancement de l'application via 'python app/main.py'...")
    # Configuration de Uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))

    # Lancer Uvicorn
    uvicorn.run("app.main:app", host=host, port=port, reload=True, log_level="info")
