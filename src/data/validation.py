import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Dict, Any
import json
from datetime import datetime
import traceback

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
    logger.setLevel(logging.DEBUG)  # Changed to DEBUG for more detailed logging
    
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
        log_dir / f"data_validation_{datetime.now().strftime('%Y%m%d')}.log",
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

class DataValidator:
    def __init__(self, config_path: str = "configs/data_schema.yaml"):
        logger.info(f"Initializing DataValidator with config: {config_path}")
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Successfully loaded config with {len(self.config)} keys")
            logger.debug(f"Config contents: {self.config}")
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading config: {e}")
            raise
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Valide le schéma des données"""
        logger.info("Starting schema validation")
        logger.debug(f"DataFrame shape: {df.shape}")
        logger.debug(f"DataFrame columns: {list(df.columns)}")
        logger.debug(f"DataFrame dtypes: {df.dtypes.to_dict()}")
        
        validation_results = {
            'schema_valid': True,
            'issues': [],
            'stats': {}
        }
        
        # Vérifier les colonnes attendues
        expected_cols = self.config.get('expected_columns', [])
        logger.info(f"Checking against {len(expected_cols)} expected columns")
        logger.debug(f"Expected columns: {expected_cols}")
        
        missing_cols = set(expected_cols) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_cols)
        
        if missing_cols:
            validation_results['schema_valid'] = False
            validation_results['issues'].append(f"Colonnes manquantes: {missing_cols}")
            logger.warning(f"Missing columns detected: {missing_cols}")
        else:
            logger.info("All expected columns are present")
        
        if extra_cols:
            validation_results['issues'].append(f"Colonnes supplémentaires: {extra_cols}")
            logger.warning(f"Extra columns detected: {extra_cols}")
        else:
            logger.info("No extra columns found")
        
        # Validate data types if specified in config
        expected_dtypes = self.config.get('expected_dtypes', {})
        if expected_dtypes:
            logger.info("Validating data types")
            for col, expected_dtype in expected_dtypes.items():
                if col in df.columns:
                    actual_dtype = str(df[col].dtype)
                    if actual_dtype != expected_dtype:
                        validation_results['issues'].append(f"Column {col}: expected {expected_dtype}, got {actual_dtype}")
                        logger.warning(f"Data type mismatch for {col}: expected {expected_dtype}, got {actual_dtype}")
                    else:
                        logger.debug(f"Data type OK for {col}: {actual_dtype}")
        
        # Check for null values in required columns
        required_cols = self.config.get('required_columns', [])
        if required_cols:
            logger.info(f"Checking {len(required_cols)} required columns for null values")
            for col in required_cols:
                if col in df.columns:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        validation_results['issues'].append(f"Required column {col} has {null_count} null values")
                        logger.error(f"Required column {col} contains {null_count} null values")
                    else:
                        logger.debug(f"Required column {col} has no null values")
        
        # Statistiques de base
        logger.info("Calculating basic statistics")
        try:
            missing_values = df.isnull().sum().to_dict()
            total_missing = sum(missing_values.values())
            logger.info(f"Total missing values across all columns: {total_missing}")
            
            validation_results['stats'] = {
                'n_rows': len(df),
                'n_cols': len(df.columns),
                'missing_values': missing_values,
                'data_types': df.dtypes.astype(str).to_dict(),
                'numeric_stats': {}
            }
            
            # Numeric statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                logger.info(f"Calculating statistics for {len(numeric_cols)} numeric columns")
                validation_results['stats']['numeric_stats'] = df.describe().to_dict()
                
                # Log some key stats for each numeric column
                for col in numeric_cols:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    min_val = df[col].min()
                    max_val = df[col].max()
                    logger.debug(f"Column {col} - Mean: {mean_val:.2f}, Std: {std_val:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f}")
            else:
                logger.info("No numeric columns found")
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            logger.debug(f"Statistics error traceback: {traceback.format_exc()}")
        
        # Validate value ranges if specified in config
        value_ranges = self.config.get('value_ranges', {})
        if value_ranges:
            logger.info("Validating value ranges")
            for col, range_config in value_ranges.items():
                if col in df.columns:
                    min_val = range_config.get('min')
                    max_val = range_config.get('max')
                    
                    if min_val is not None:
                        below_min = (df[col] < min_val).sum()
                        if below_min > 0:
                            validation_results['issues'].append(f"Column {col}: {below_min} values below minimum {min_val}")
                            logger.warning(f"Column {col}: {below_min} values below minimum {min_val}")
                    
                    if max_val is not None:
                        above_max = (df[col] > max_val).sum()
                        if above_max > 0:
                            validation_results['issues'].append(f"Column {col}: {above_max} values above maximum {max_val}")
                            logger.warning(f"Column {col}: {above_max} values above maximum {max_val}")
        
        logger.info(f"Schema validation completed. Valid: {validation_results['schema_valid']}, Issues: {len(validation_results['issues'])}")
        return validation_results
    
    def detect_drift(self, current_df: pd.DataFrame, reference_stats: Dict = None) -> Dict[str, Any]:
        """Détecte la dérive des données"""
        logger.info("Starting drift detection")
        
        drift_results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'column_drifts': {}
        }
        
        if reference_stats is None:
            logger.warning("No reference statistics provided for drift detection")
            return drift_results
        
        logger.debug(f"Reference stats keys: {list(reference_stats.keys())}")
        
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns
        logger.info(f"Analyzing drift for {len(numeric_cols)} numeric columns")
        
        drift_threshold = self.config.get('drift_threshold', 0.1)
        logger.debug(f"Using drift threshold: {drift_threshold}")
        
        for col in numeric_cols:
            if col in reference_stats.get('numeric_stats', {}):
                try:
                    current_mean = current_df[col].mean()
                    reference_mean = reference_stats['numeric_stats'][col]['mean']
                    
                    logger.debug(f"Column {col} - Current mean: {current_mean:.4f}, Reference mean: {reference_mean:.4f}")
                    
                    # Simple drift detection basé sur la différence de moyenne
                    drift_score = abs(current_mean - reference_mean) / reference_mean if reference_mean != 0 else 0
                    drift_results['column_drifts'][col] = drift_score
                    
                    logger.debug(f"Column {col} drift score: {drift_score:.4f}")
                    
                    if drift_score > drift_threshold:
                        drift_results['drift_detected'] = True
                        logger.warning(f"Drift detected in column {col}: score {drift_score:.4f} > threshold {drift_threshold}")
                    else:
                        logger.debug(f"No significant drift in column {col}")
                        
                except Exception as e:
                    logger.error(f"Error calculating drift for column {col}: {e}")
            else:
                logger.warning(f"No reference statistics available for column {col}")
        
        if drift_results['column_drifts']:
            drift_results['drift_score'] = np.mean(list(drift_results['column_drifts'].values()))
            logger.info(f"Overall drift score: {drift_results['drift_score']:.4f}")
        else:
            logger.warning("No drift scores calculated")
        
        logger.info(f"Drift detection completed. Drift detected: {drift_results['drift_detected']}")
        return drift_results
    
    def run_validation(self, data_path: str) -> Dict[str, Any]:
        """Exécute la validation complète"""
        logger.info(f"Starting complete validation for: {data_path}")
        
        try:
            # Check if file exists
            if not Path(data_path).exists():
                logger.error(f"Data file not found: {data_path}")
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            logger.info(f"Loading data from: {data_path}")
            file_size = Path(data_path).stat().st_size
            logger.debug(f"File size: {file_size} bytes")
            
            # Load data with error handling
            try:
                df = pd.read_csv(data_path)
                logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            except pd.errors.EmptyDataError:
                logger.error("The CSV file is empty")
                raise
            except pd.errors.ParserError as e:
                logger.error(f"Error parsing CSV file: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error loading CSV: {e}")
                raise
            
            # Validation du schéma
            logger.info("Running schema validation")
            schema_results = self.validate_schema(df)
            
            # Try to load reference stats for drift detection
            try:
                reference_path = Path("data/validation/reference_stats.json")
                if reference_path.exists():
                    logger.info("Loading reference statistics for drift detection")
                    with open(reference_path, 'r') as f:
                        reference_stats = json.load(f)
                    
                    drift_results = self.detect_drift(df, reference_stats)
                    schema_results['drift_analysis'] = drift_results
                else:
                    logger.info("No reference statistics found, skipping drift detection")
            except Exception as e:
                logger.warning(f"Could not perform drift analysis: {e}")
            
            # Sauvegarde des résultats
            output_dir = Path("data/validation")
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving validation results to: {output_dir}")
            
            # Save validation results
            results_file = output_dir / "validation_results.json"
            with open(results_file, 'w') as f:
                json.dump(schema_results, f, indent=2, default=str)
            logger.info(f"Validation results saved to: {results_file}")
            
            # Save current stats as reference for future drift detection
            if schema_results.get('stats'):
                reference_file = output_dir / "reference_stats.json"
                with open(reference_file, 'w') as f:
                    json.dump(schema_results['stats'], f, indent=2, default=str)
                logger.info(f"Reference statistics saved to: {reference_file}")
            
            logger.info(f"Validation completed successfully. Schema valid: {schema_results['schema_valid']}")
            if schema_results['issues']:
                logger.warning(f"Validation completed with {len(schema_results['issues'])} issues")
                for issue in schema_results['issues']:
                    logger.warning(f"Issue: {issue}")
            
            return schema_results
            
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            logger.debug(f"Full error traceback: {traceback.format_exc()}")
            raise

if __name__ == "__main__":
    logger.info("Starting DataValidator execution")
    try:
        validator = DataValidator()
        results = validator.run_validation("/opt/airflow/data/raw/synthetic_data.csv")
        logger.info("DataValidator execution completed successfully")
    except Exception as e:
        logger.error(f"DataValidator execution failed: {e}")
        