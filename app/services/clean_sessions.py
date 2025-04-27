"""
Module pour nettoyer les sessions et tokens Google existants.
"""
import os
import shutil
import logging

def clean_google_tokens():
    """
    Supprime tous les tokens Google et fichiers de session existants.
    """
    try:
        # Chemin du répertoire de tokens
        app_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        token_dir = os.path.join(app_root, 'tokens')
        
        # Supprimer les fichiers de token s'ils existent
        if os.path.exists(token_dir):
            for file in os.listdir(token_dir):
                if file.endswith('.json') or 'token' in file:
                    os.remove(os.path.join(token_dir, file))
                    logging.info(f"Fichier de token supprimé: {file}")
        
        # Supprimer les fichiers de session s'ils existent
        session_dir = os.path.join(app_root, 'sessions')
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            os.makedirs(session_dir, exist_ok=True)
            logging.info("Répertoire de sessions nettoyé")
        
        logging.info("Nettoyage des tokens et sessions terminé avec succès")
        return True
    
    except Exception as e:
        logging.error(f"Erreur lors du nettoyage des tokens: {str(e)}")
        return False

if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Exécuter le nettoyage
    clean_google_tokens()
