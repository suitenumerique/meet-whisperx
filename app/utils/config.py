import os

APP_VERSION = os.getenv("APP_VERSION")
API_KEY = os.getenv("API_KEY")
TIMEOUT_KEEP_ALIVE = int(os.getenv("TIMEOUT_KEEP_ALIVE", 60))