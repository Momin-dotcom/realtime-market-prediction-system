from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '/opt/airflow')

from src.ingest.yahoo_scraper import fetch_yahoo_data
from src.ingest.rss_scraper import fetch_rss_articles
from src.ingest.reddit_scraper import fetch_reddit_posts
from src.ingest.twitter_scraper import fetch_tweets
from src.sentiment.labeler import label_all_sources
from src.timeseries.builder import build_timeseries

default_args = {
    'owner': 'member1',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
}

with DAG(
    dag_id='market_prediction_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    description='Ingest → Sentiment Label → Build Timeseries',
    tags=['market', 'sentiment', 'timeseries'],
) as dag:

    # ── Task 1a: ingest price data (yahoo) ───────────────────────
    ingest_yahoo = PythonOperator(
        task_id='ingest_yahoo',
        python_callable=fetch_yahoo_data,
    )

    # ── Task 1b: ingest text sources ────────────────────────────
    ingest_rss = PythonOperator(
        task_id='ingest_rss',
        python_callable=fetch_rss_articles,
    )

    ingest_reddit = PythonOperator(
        task_id='ingest_reddit',
        python_callable=fetch_reddit_posts,
    )

    ingest_twitter = PythonOperator(
        task_id='ingest_twitter',
        python_callable=fetch_tweets,
    )

    # ── Task 2: label sentiment (text sources only) ──────────────
    label_sentiment = PythonOperator(
        task_id='label_sentiment',
        python_callable=label_all_sources,
    )

    # ── Task 3: build timeseries (price + sentiment merged) ──────
    build_ts = PythonOperator(
        task_id='build_timeseries',
        python_callable=build_timeseries,
    )

    # ── dependencies ─────────────────────────────────────────────
    # text sources → sentiment labeling
    [ingest_rss, ingest_reddit, ingest_twitter] >> label_sentiment

    # yahoo (price) + labeled sentiment → timeseries builder
    [ingest_yahoo, label_sentiment] >> build_ts