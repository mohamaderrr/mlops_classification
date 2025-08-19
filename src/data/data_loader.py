import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import sys
import time
from datetime import datetime
from sklearn.datasets import make_classification

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
        log_dir / f"data_loader_{datetime.now().strftime('%Y%m%d')}.log",
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

class DataLoader:
    def __init__(self, config_path: str = "configs/data_schema.yaml"):
        """Initialize DataLoader with configuration"""
        logger.info("🚀 Initializing DataLoader")
        logger.info(f"📄 Loading configuration from: {config_path}")
        
        start_time = time.time()
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            load_time = time.time() - start_time
            logger.info(f"✅ Configuration loaded successfully in {load_time:.3f}s")
            logger.debug(f"📋 Config keys: {list(self.config.keys())}")
            
        except FileNotFoundError:
            logger.error(f"❌ Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error loading config: {e}")
            raise

    
    def generate_synthetic_data(self) -> tuple:
        """Generate synthetic classification data"""
        logger.info("🎲 Generating synthetic classification data...")
        
        # Default parameters
        params = {
            'n_samples': self.config.get('n_samples', 10000),
            'n_features': self.config.get('n_features', 20),
            'n_informative': self.config.get('n_informative', 15),
            'n_redundant': self.config.get('n_redundant', 5),
            'n_classes': self.config.get('n_classes', 3),
            'random_state': self.config.get('random_state', 42)
        }
        
        logger.info(f"📊 Data generation parameters:")
        for key, value in params.items():
            logger.info(f"   • {key}: {value}")
        
        start_time = time.time()
        try:
            X, y = make_classification(**params)
            generation_time = time.time() - start_time
            
            logger.info(f"✅ Synthetic data generated successfully in {generation_time:.3f}s")
            logger.info(f"📏 Features shape: {X.shape}")
            logger.info(f"🎯 Target shape: {y.shape}")
            logger.info(f"🏷️  Target classes: {np.unique(y)}")
            logger.info(f"📈 Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Error generating synthetic data: {e}")
            raise

    def create_dataframe(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Create pandas DataFrame from features and target"""
        logger.info("🏗️  Creating DataFrame...")
        
        start_time = time.time()
        try:
            # Create feature column names
            feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
            logger.debug(f"📝 Feature columns: {feature_cols[:5]}... (showing first 5)")
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=feature_cols)
            df['target'] = y
            
            creation_time = time.time() - start_time
            logger.info(f"✅ DataFrame created successfully in {creation_time:.3f}s")
            logger.info(f"📊 Final DataFrame shape: {df.shape}")
            logger.info(f"🗂️  Columns: {len(df.columns)} total ({len(feature_cols)} features + 1 target)")
            
            # Data quality checks
            logger.info("🔍 Performing data quality checks...")
            
            # Check for missing values
            missing_values = df.isnull().sum().sum()
            logger.info(f"❓ Missing values: {missing_values}")
            
            # Check for duplicates
            duplicates = df.duplicated().sum()
            logger.info(f"🔄 Duplicate rows: {duplicates}")
            
            # Memory usage
            memory_mb = df.memory_usage(deep=True).sum() /  sys.exit(1)1024 / 1024
            logger.info(f"💾 Memory usage: {memory_mb:.2f} MB")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating DataFrame: {e}")
            raise

    def save_data(self, df: pd.DataFrame) -> Path:
        """Save DataFrame to configured path"""
        logger.info("💾 Saving data to file...")
        
        try:
            output_path = Path(self.config['raw_data_path'])
            logger.info(f"📁 Output path: {output_path}")
            
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"📂 Created directory: {output_path.parent}")
            
            # Save with timing
            start_time = time.time()
            df.to_csv(output_path, index=False)
            save_time = time.time() - start_time
            
            # Verify file was created
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / 1024 / 1024
                logger.info(f"✅ Data saved successfully in {save_time:.3f}s")
                logger.info(f"📄 File size: {file_size_mb:.2f} MB")
                logger.info(f"🔗 Full path: {output_path.absolute()}")
            else:
                logger.error("❌ File was not created successfully")
                raise FileNotFoundError(f"Output file not found: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error saving data: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """Main method to load/generate and save data"""
        logger.info("=" * 60)
        logger.info("🎯 STARTING DATA LOADING PROCESS")
        logger.info("=" * 60)
        
        total_start_time = time.time()
        
        try:
            # Step 1: Validate configuration
            self.validate_config()
            
            # Step 2: Generate synthetic data
            X, y = self.generate_synthetic_data()
            
            # Step 3: Create DataFrame
            df = self.create_dataframe(X, y)
            
            # Step 4: Save data
            output_path = self.save_data(df)
            
            # Final summary
            total_time = time.time() - total_start_time
            logger.info("=" * 60)
            logger.info("🎉 DATA LOADING COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"⏱️  Total execution time: {total_time:.3f}s")
            logger.info(f"📊 Final dataset shape: {df.shape}")
            logger.info(f"📁 Saved to: {output_path}")
            logger.info("=" * 60)
            
            return df
            
        except Exception as e:
            total_time = time.time() - total_start_time
            logger.error("=" * 60)
            logger.error("💥 DATA LOADING FAILED")
            logger.error("=" * 60)
            logger.error(f"❌ Error: {str(e)}")
            logger.error(f"⏱️  Failed after: {total_time:.3f}s")
            logger.error("=" * 60)
            raise

if __name__ == "__main__":
    logger.info("🔧 Running DataLoader as standalone script")
    
    try:
        loader = DataLoader()
        data = loader.load_data()
        logger.info("🏁 Script execution completed successfully")
        
    except Exception as e:
        logger.error(f"💥 Script failed with error: {e}")
       