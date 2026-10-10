"""
Data Ingestion and Loading Module for Telco Customer Churn.
Downloads and verifies the raw dataset into data/WA_Fn-UseC_-Telco-Customer-Churn.csv.
"""
from pathlib import Path
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def download_raw_data(force: bool = False) -> Path:
    """Download the raw Telco Customer Churn dataset if not already present."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    if DATA_FILE.exists() and not force:
        logger.info(f"Dataset already present at: {DATA_FILE}")
        return DATA_FILE
    
    logger.info(f"Downloading dataset from {RAW_DATA_URL}...")
    urllib.request.urlretrieve(RAW_DATA_URL, DATA_FILE)
    logger.info(f"Successfully saved to: {DATA_FILE}")
    return DATA_FILE


if __name__ == "__main__":
    download_raw_data()
