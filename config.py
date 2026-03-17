# config.py - all parameters at one place

BBOX = [56.35783614321667, 25.24184438049558, 57.288912900645954, 26.66044443177684]
DATE = "2026-03-16"

TILE_SIZE = 640
OVERLAP = 64
CONF_THRESHOLD = 0.25
NMS_THRESHOLD_DEG = 0.01

MODEL_REPO = "hewitleo/sar-ship-detection-yolov8"
MODEL_FILE = "weights_(model)/best.pt"

OUTPUT_DIR = "./outputs"
TIFF_CACHE_DIR = "./cache"