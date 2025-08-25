from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import logging
import requests
from airflow.models import Variable

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,   # <-- ensures re-run each time
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


dag = DAG(
    'ml_classification_pipeline',
    default_args=default_args,
    description='Pipeline MLOps pour classification',
    schedule_interval='* * * * *',   # <-- run every 1 minute
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'classification', 'production']
)


# --- Step 1: Load Data ---
load_data = BashOperator(
    task_id='load_data',
    bash_command='cd /opt/airflow && python src/data/data_loader.py',
    dag=dag,
)

# --- Step 2: Validate Data ---
validate_data = BashOperator(
    task_id='validate_data',
    bash_command='cd /opt/airflow && python src/data/validation.py',
    dag=dag,
)

# --- Step 3: Validation Gate ---
def check_validation_results(**context):
    import json
    try:
        with open('/opt/airflow/data/validation/validation_results.json', 'r') as f:
            results = json.load(f)
        if not results['schema_valid']:
            raise ValueError(f"Validation échouée: {results['issues']}")
        logger.info("Validation des données réussie")
        return True
    except Exception as e:
        logger.error(f"Erreur validation: {e}")
        raise

validation_gate = PythonOperator(
    task_id='validation_gate',
    python_callable=check_validation_results,
    dag=dag,
)

# --- Step 4: Preprocess ---
preprocess_data = BashOperator(
    task_id='preprocess_data',
    bash_command='cd /opt/airflow && python src/features/preprocessing.py',
    dag=dag,
)

# --- Step 5: Train Model ---
train_model = BashOperator(
    task_id='train_model',
    bash_command='cd /opt/airflow && python src/models/train.py',
    dag=dag,
)




# --- Dependencies ---
load_data >> validate_data >> validation_gate >> preprocess_data
preprocess_data >> train_model 
