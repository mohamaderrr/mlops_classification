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
    logger.info(f"Created/verified logs directory: {log_dir.absolute()}")
    
    file_handler = logging.FileHandler(
        log_dir / f"preprocessing_{datetime.now().strftime('%Y%m%d')}.log",
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("Logging system initialized successfully")
    return logger

# Initialize logger
logger = setup_logging()


class DataPreprocessor:
    def __init__(self, config_path: str = "params.yaml"):
        """Initialize DataPreprocessor with configuration"""
        logger.info(f"Initializing DataPreprocessor with config: {config_path}")
        
        try:
            # Load configuration
            if Path(config_path).exists():
                logger.info(f"Loading configuration from {config_path}")
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Configuration loaded successfully: {self.config}")
            else:
                logger.warning(f"Configuration file {config_path} not found, using default values")
                self.config = {
                    'test_size': 0.2,
                    'random_state': 42,
                    'scale_features': True
                }
                logger.info(f"Using default configuration: {self.config}")
        
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            logger.info("Falling back to default configuration")
            self.config = {
                'test_size': 0.2,
                'random_state': 42,
                'scale_features': True
            }
        
        # Initialize preprocessing components
        logger.info("Initializing StandardScaler")
        self.scaler = StandardScaler()
        
        logger.info("Initializing LabelEncoder")
        self.label_encoder = LabelEncoder()
        
        logger.info("DataPreprocessor initialization completed")
    
    def preprocess_data(self, data_path: str):
        """Preprocess data for training with comprehensive logging"""
        logger.info(f"Starting data preprocessing for: {data_path}")
        
        try:
            # Load data
            logger.info(f"Loading dataset from: {data_path}")
            if not Path(data_path).exists():
                logger.error(f"Data file not found: {data_path}")
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            df = pd.read_csv(data_path)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            logger.info(f"Dataset columns: {df.columns.tolist()}")
            logger.info(f"Dataset memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Log basic statistics
            logger.info(f"Dataset info:")
            logger.info(f"  - Number of rows: {len(df)}")
            logger.info(f"  - Number of columns: {len(df.columns)}")
            logger.info(f"  - Missing values per column: {df.isnull().sum().to_dict()}")
            logger.info(f"  - Data types: {df.dtypes.to_dict()}")
            
            # Check for target column
            target_col = 'target'
            if target_col not in df.columns:
                logger.error(f"Target column '{target_col}' not found in dataset")
                logger.info(f"Available columns: {df.columns.tolist()}")
                raise ValueError(f"Target column '{target_col}' not found")
            
            logger.info(f"Target column '{target_col}' found")
            
            # Separate features and target
            logger.info("Separating features and target variable")
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            
            logger.info(f"Features shape: {X.shape}")
            logger.info(f"Target shape: {y.shape}")
            logger.info(f"Feature columns: {X.columns.tolist()}")
            
            # Analyze target variable
            logger.info("Analyzing target variable:")
            logger.info(f"  - Target data type: {y.dtype}")
            logger.info(f"  - Unique values: {y.nunique()}")
            logger.info(f"  - Value counts: {y.value_counts().to_dict()}")
            
            # Handle target encoding if necessary
            if y.dtype == 'object':
                logger.info("Target variable is categorical, applying label encoding")
                original_values = y.unique()
                y = self.label_encoder.fit_transform(y)
                encoded_mapping = dict(zip(original_values, self.label_encoder.transform(original_values)))
                logger.info(f"Label encoding mapping: {encoded_mapping}")
                
                # Save label encoder mapping for future reference
                logger.info("Saving label encoder for future use")
            else:
                logger.info("Target variable is numerical, no encoding needed")
            
            # Check class balance
            unique_vals, counts = np.unique(y, return_counts=True)
            class_distribution = dict(zip(unique_vals, counts))
            logger.info(f"Class distribution: {class_distribution}")
            
            # Calculate class balance ratio
            min_class_count = min(counts)
            max_class_count = max(counts)
            balance_ratio = min_class_count / max_class_count
            logger.info(f"Class balance ratio (min/max): {balance_ratio:.3f}")
            
            if balance_ratio < 0.1:
                logger.warning("Dataset appears to be highly imbalanced (ratio < 0.1)")
            elif balance_ratio < 0.5:
                logger.warning("Dataset appears to be moderately imbalanced (ratio < 0.5)")
            else:
                logger.info("Dataset appears to be reasonably balanced")
            
            # Train-test split
            logger.info(f"Performing train-test split with test_size={self.config['test_size']}")
            logger.info(f"Using random_state={self.config['random_state']} for reproducibility")
            logger.info("Using stratified split to maintain class distribution")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['test_size'],
                random_state=self.config['random_state'],
                stratify=y
            )
            
            logger.info(f"Train-test split completed:")
            logger.info(f"  - Training set: {X_train.shape}")
            logger.info(f"  - Test set: {X_test.shape}")
            logger.info(f"  - Training target distribution: {np.bincount(y_train)}")
            logger.info(f"  - Test target distribution: {np.bincount(y_test)}")
            
            # Feature scaling
            if self.config.get('scale_features', True):
                logger.info("Applying feature scaling using StandardScaler")
                logger.info("Fitting scaler on training data")
                
                # Log feature statistics before scaling
                logger.info("Feature statistics before scaling:")
                logger.info(f"  - Training features mean: {X_train.mean().to_dict()}")
                logger.info(f"  - Training features std: {X_train.std().to_dict()}")
                
                X_train_scaled = self.scaler.fit_transform(X_train)
                logger.info("Scaler fitted on training data")
                
                logger.info("Transforming test data using fitted scaler")
                X_test_scaled = self.scaler.transform(X_test)
                
                # Log feature statistics after scaling
                logger.info("Feature statistics after scaling:")
                logger.info(f"  - Training features mean: {np.mean(X_train_scaled, axis=0)}")
                logger.info(f"  - Training features std: {np.std(X_train_scaled, axis=0)}")
                logger.info(f"  - Test features mean: {np.mean(X_test_scaled, axis=0)}")
                logger.info(f"  - Test features std: {np.std(X_test_scaled, axis=0)}")
                
            else:
                logger.info("Feature scaling disabled, using original values")
                X_train_scaled = X_train.values
                X_test_scaled = X_test.values
            
            logger.info("Feature preprocessing completed")
            
            # Create output directory
            output_dir = Path("data/processed")
            logger.info(f"Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory ready: {output_dir.absolute()}")
            
            # Prepare data for saving
            logger.info("Preparing training data for saving")
            train_data = {
                'X': X_train_scaled,
                'y': y_train.values,
                'feature_names': X.columns.tolist()
            }
            
            logger.info("Preparing test data for saving")
            test_data = {
                'X': X_test_scaled,
                'y': y_test.values,
                'feature_names': X.columns.tolist()
            }
            
            # Save training data
            train_path = output_dir / "train.pkl"
            logger.info(f"Saving training data to: {train_path}")
            try:
                with open(train_path, 'wb') as f:
                    pickle.dump(train_data, f)
                logger.info(f"Training data saved successfully. File size: {train_path.stat().st_size / 1024**2:.2f} MB")
            except Exception as e:
                logger.error(f"Error saving training data: {str(e)}")
                raise
            
            # Save test data
            test_path = output_dir / "test.pkl"
            logger.info(f"Saving test data to: {test_path}")
            try:
                with open(test_path, 'wb') as f:
                    pickle.dump(test_data, f)
                logger.info(f"Test data saved successfully. File size: {test_path.stat().st_size / 1024**2:.2f} MB")
            except Exception as e:
                logger.error(f"Error saving test data: {str(e)}")
                raise
            
            # Save scaler
            if self.config.get('scale_features', True):
                scaler_path = output_dir / "scaler.pkl"
                logger.info(f"Saving scaler to: {scaler_path}")
                try:
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scaler, f)
                    logger.info(f"Scaler saved successfully. File size: {scaler_path.stat().st_size / 1024:.2f} KB")
                except Exception as e:
                    logger.error(f"Error saving scaler: {str(e)}")
                    raise
            
            # Save label encoder if used
            if y.dtype != df[target_col].dtype:  # If encoding was applied
                encoder_path = output_dir / "label_encoder.pkl"
                logger.info(f"Saving label encoder to: {encoder_path}")
                try:
                    with open(encoder_path, 'wb') as f:
                        pickle.dump(self.label_encoder, f)
                    logger.info(f"Label encoder saved successfully")
                except Exception as e:
                    logger.error(f"Error saving label encoder: {str(e)}")
                    raise
            
            # Log final summary
            logger.info("="*50)
            logger.info("PREPROCESSING SUMMARY")
            logger.info("="*50)
            logger.info(f"✓ Dataset loaded: {df.shape}")
            logger.info(f"✓ Features: {X.shape[1]} columns")
            logger.info(f"✓ Training samples: {X_train_scaled.shape[0]}")
            logger.info(f"✓ Test samples: {X_test_scaled.shape[0]}")
            logger.info(f"✓ Number of classes: {len(np.unique(y))}")
            logger.info(f"✓ Feature scaling: {'Enabled' if self.config.get('scale_features', True) else 'Disabled'}")
            logger.info(f"✓ Label encoding: {'Applied' if y.dtype != df[target_col].dtype else 'Not needed'}")
            logger.info(f"✓ Files saved to: {output_dir.absolute()}")
            logger.info("="*50)
            logger.info("Preprocessing completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise


if __name__ == "__main__":
    logger.info("Starting data preprocessing script")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    
    try:
        preprocessor = DataPreprocessor()
        preprocessor.preprocess_data("/opt/airflow/data/raw/synthetic_data.csv")
        logger.info("Script completed successfully")
    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
     

    