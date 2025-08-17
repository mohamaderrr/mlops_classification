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
    'depends_on_past': False,
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
    schedule_interval='@daily',
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

# --- Step 6: Evaluate ---
evaluate_model = BashOperator(
    task_id='evaluate_model',
    bash_command='cd /opt/airflow && python src/models/evaluate.py',
    dag=dag,
)

# --- Step 7: Performance Gate ---
def check_model_performance(**context):
    import json
    try:
        with open('/opt/airflow/reports/evaluation.json', 'r') as f:
            metrics = json.load(f)
        min_accuracy = 0.8
        if metrics['accuracy'] < min_accuracy:
            raise ValueError(f"Performance insuffisante: {metrics['accuracy']:.3f} < {min_accuracy}")
        logger.info(f"Performance acceptable: {metrics['accuracy']:.3f}")
        return True
    except Exception as e:
        logger.error(f"Erreur performance: {e}")
        raise

performance_gate = PythonOperator(
    task_id='performance_gate',
    python_callable=check_model_performance,
    dag=dag,
)

# --- Step 8: Trigger GitHub Actions CI/CD ---
def trigger_github_action(**context):
    token = Variable.get("GITHUB_TOKEN")  # store GitHub PAT in Airflow Variables
    repo = Variable.get("GITHUB_REPO", default_var="your-username/mlops-classification-project")
    tag = context['ds']  # use Airflow execution date as tag

    url = f"https://api.github.com/repos/{repo}/actions/workflows/deploy.yml/dispatches"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
    }
    data = {"ref": "main", "inputs": {"tag": tag}}

    r = requests.post(url, headers=headers, json=data)
    if r.status_code != 204:
        raise Exception(f"Trigger GitHub Action failed: {r.text}")
    logger.info(f"Triggered GitHub Action for tag {tag}")

trigger_ci_cd = PythonOperator(
    task_id="trigger_ci_cd",
    python_callable=trigger_github_action,
    provide_context=True,
    dag=dag,
)

# --- Dependencies ---
load_data >> validate_data >> validation_gate >> preprocess_data
preprocess_data >> train_model >> evaluate_model >> performance_gate 
