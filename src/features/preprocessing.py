import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import pickle
import yaml
import logging
import sys
from datetime import datetime
# Configure logging with detailed format
def setup_logging():
    """Configure logging with custom format and multiple handlers"""
    
    # Create custom formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler for persistent logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(
        log_dir / f"preprocing_{datetime.now().strftime('%Y%m%d')}.log",
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()


class DataPreprocessor:
    def __init__(self, config_path: str = "params.yaml"):
    
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def preprocess_data(self, data_path: str):
        """Prétraite les données pour l'entraînement"""
        df = pd.read_csv(data_path)
        
        # Séparation features/target
        target_col = 'target'
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Encodage du target si nécessaire
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Normalisation des features
        if self.config.get('scale_features', True):
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Sauvegarde des données prétraitées
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder les données
        train_data = {
            'X': X_train_scaled,
            'y': y_train.values,
            'feature_names': X.columns.tolist()
        }
        
        test_data = {
            'X': X_test_scaled,
            'y': y_test.values,
            'feature_names': X.columns.tolist()
        }
        
        with open(output_dir / "train.pkl", 'wb') as f:
            pickle.dump(train_data, f)
        
        with open(output_dir / "test.pkl", 'wb') as f:
            pickle.dump(test_data, f)
        
        # Sauvegarder le scaler
        with open(output_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Prétraitement terminé:")
        logger.info(f"Train shape: {X_train_scaled.shape}")
        logger.info(f"Test shape: {X_test_scaled.shape}")
        logger.info(f"Nombre de classes: {len(np.unique(y))}")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_data("data/raw/dataset.csv")