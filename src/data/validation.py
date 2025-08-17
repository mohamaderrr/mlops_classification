import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Dict, Any
import json

logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, config_path: str = "configs/data_schema.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Valide le schéma des données"""
        validation_results = {
            'schema_valid': True,
            'issues': [],
            'stats': {}
        }
        
        # Vérifier les colonnes attendues
        expected_cols = self.config.get('expected_columns', [])
        missing_cols = set(expected_cols) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_cols)
        
        if missing_cols:
            validation_results['schema_valid'] = False
            validation_results['issues'].append(f"Colonnes manquantes: {missing_cols}")
        
        if extra_cols:
            validation_results['issues'].append(f"Colonnes supplémentaires: {extra_cols}")
        
        # Statistiques de base
        validation_results['stats'] = {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'numeric_stats': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        return validation_results
    
    def detect_drift(self, current_df: pd.DataFrame, reference_stats: Dict = None) -> Dict[str, Any]:
        """Détecte la dérive des données"""
        drift_results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'column_drifts': {}
        }
        
        if reference_stats is None:
            return drift_results
        
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in reference_stats.get('numeric_stats', {}):
                current_mean = current_df[col].mean()
                reference_mean = reference_stats['numeric_stats'][col]['mean']
                
                # Simple drift detection basé sur la différence de moyenne
                drift_score = abs(current_mean - reference_mean) / reference_mean if reference_mean != 0 else 0
                drift_results['column_drifts'][col] = drift_score
                
                if drift_score > self.config.get('drift_threshold', 0.1):
                    drift_results['drift_detected'] = True
        
        drift_results['drift_score'] = np.mean(list(drift_results['column_drifts'].values())) if drift_results['column_drifts'] else 0
        
        return drift_results
    
    def run_validation(self, data_path: str) -> Dict[str, Any]:
        """Exécute la validation complète"""
        df = pd.read_csv(data_path)
        
        # Validation du schéma
        schema_results = self.validate_schema(df)
        
        # Sauvegarde des résultats
        output_dir = Path("data/validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "validation_results.json", 'w') as f:
            json.dump(schema_results, f, indent=2)
        
        logger.info(f"Validation terminée. Schema valide: {schema_results['schema_valid']}")
        
        return schema_results

if __name__ == "__main__":
    validator = DataValidator()
    results = validator.run_validation("data/raw/dataset.csv")