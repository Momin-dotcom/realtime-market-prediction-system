# dags/market_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/opt/airflow')

from src.ingest.yahoo_scraper import fetch_yahoo_data
from src.ingest.rss_scraper import fetch_rss_articles
from src.ingest.reddit_scraper import fetch_reddit_posts
from src.sentiment.labeler import label_all_sources
from src.timeseries.builder import build_timeseries

default_args = {
    'owner': 'member1',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

with DAG(
    dag_id='market_prediction_pipeline',
    default_args=default_args,
    schedule_interval='@hourly',
    catchup=False,
    description='Ingest → Label → Build timeseries'
) as dag:

    ingest = PythonOperator(
        task_id='ingest_data',
        python_callable=lambda: [
            fetch_yahoo_data(),
            fetch_rss_articles(),
            fetch_reddit_posts(),
        ]
    )

    label = PythonOperator(
        task_id='label_sentiment',
        python_callable=label_all_sources
    )

    build = PythonOperator(
        task_id='build_timeseries',
        python_callable=build_timeseries
    )

    ingest >> label >> build