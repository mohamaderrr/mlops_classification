import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from sklearn.datasets import make_classification

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config_path: str = "configs/data_schema.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_data(self) -> pd.DataFrame:
        """Charge les données depuis la source configurée"""
        try:
            # Pour cet exemple, on génère des données synthétiques
            # En production, remplacer par le chargement réel
            X, y = make_classification(
                n_samples=10000,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=3,
                random_state=42
            )
            
            # Créer un DataFrame avec les features et target
            feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_cols)
            df['target'] = y
            
            # Sauvegarder
            output_path = Path(self.config['raw_data_path'])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Données sauvegardées: {output_path}")
            logger.info(f"Shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            raise

if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_data()