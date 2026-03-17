# pipeline.py

import argparse
import json
import os
import numpy as np
import cv2
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from pathlib import Path
from shapely.geometry import box as shapely_box
import geopandas as gpd
import planetary_computer
import pystac_client
import odc.stac
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

import config


def search_and_download(bbox, date):
    """Search and download Sentinel-1 scenes from Planetary Computer."""
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Search for scenes on given date
    date_range = f"{date}T00:00:00Z/{date}T23:59:59Z"
    search = catalog.search(
        collections=["sentinel-1-grd"],
        bbox=bbox,
        datetime=date_range,
    )

    items = list(search.items())
    if not items:
        raise RuntimeError(f"No Sentinel-1 scenes found for {date} in given bbox.")

    print(f"Found {len(items)} scene(s)")

    # Sign all items
    signed_items = [planetary_computer.sign(item) for item in items]

    # Load and merge VV band
    ds = odc.stac.load(
        signed_items,
        bands=["vv"],
        bbox=bbox,
        resolution=0.0001,
        crs="EPSG:4326",
    )

    # Merge multiple scenes via max (fills NaN gaps)
    vv_merged = ds["vv"].max(dim="time").values
    print(f"Downloaded scene shape: {vv_merged.shape}")

    return vv_merged


def create_sea_mask(vv_merged, bbox):
    """Create sea mask from Natural Earth land polygons."""

    shp_dir = "/tmp/ne_10m_land"
    shp_path = f"{shp_dir}/ne_10m_land.shp"

    # Download Natural Earth land polygons if not cached
    if not os.path.exists(shp_dir):
        import urllib.request
        import zipfile
        url = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip"
        zip_path = "/tmp/ne_10m_land.zip"
        print("Downloading Natural Earth land mask...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(shp_dir)

    land = gpd.read_file(shp_path)
    bbox_geom = shapely_box(*bbox)
    land_clip = land[land.intersects(bbox_geom)]

    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3],
                            vv_merged.shape[1], vv_merged.shape[0])

    land_mask = rasterize(
        [(geom, 1) for geom in land_clip.geometry],
        out_shape=vv_merged.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    sea_mask = land_mask == 0
    print(f"Sea mask created, sea pixels: {sea_mask.sum():,}")

    return sea_mask


def check_bbox_size(bbox):
    """Warn user if bbox is too large and may cause memory issues."""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height

    # Rough estimate: 1 degree^2 ~ 1GB RAM at 0.0001 resolution
    estimated_gb = area * 1.0

    if estimated_gb > 4:
        print(f"WARNING: bbox area is {area:.1f} deg^2, estimated RAM usage: ~{estimated_gb:.0f}GB")
        print("Consider using a smaller bbox or increase resolution in config.py")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            raise SystemExit("Aborted by user.")
    else:
        print(f"Bbox area: {area:.2f} deg^2, estimated RAM: ~{estimated_gb:.1f}GB - OK")


def run_yolo(vv_clean, sea_mask, model, bbox):
    """Run YOLO inference on image tiles and detect ships."""

    TILE_SIZE = config.TILE_SIZE
    OVERLAP = config.OVERLAP

    all_detections = []
    rows = range(0, vv_clean.shape[0] - TILE_SIZE, TILE_SIZE - OVERLAP)
    cols = range(0, vv_clean.shape[1] - TILE_SIZE, TILE_SIZE - OVERLAP)
    total = len(rows) * len(cols)

    print(f"Running YOLO on {total} tiles...")

    for i, row in enumerate(rows):
        for col in cols:
            tile = vv_clean[row:row+TILE_SIZE, col:col+TILE_SIZE]

            # Skip empty tiles
            if np.mean(tile) < 1:
                continue

            # Normalize to 0-255 using percentiles
            p2 = np.percentile(tile, 2)
            p98 = np.percentile(tile, 98)
            tile_norm = np.clip((tile - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
            tile_rgb = cv2.merge([tile_norm, tile_norm, tile_norm])

            # Inference
            results = model(tile_rgb, conf=config.CONF_THRESHOLD, verbose=False)

            # Convert pixel coords to geo coords
            for box in results[0].boxes:
                cx = col + float(box.xywh[0][0])
                cy = row + float(box.xywh[0][1])
                lon = bbox[0] + cx / vv_clean.shape[1] * (bbox[2] - bbox[0])
                lat = bbox[3] - cy / vv_clean.shape[0] * (bbox[3] - bbox[1])
                conf = float(box.conf[0])
                all_detections.append((lon, lat, conf))

        # Progress print every 5 rows
        if i % 5 == 0:
            print(f"  Row {i+1}/{len(rows)}")

    print(f"Raw detections: {len(all_detections)}")
    return all_detections


def deduplicate(detections, sea_mask, bbox, vv_shape):
    """Remove duplicate detections from overlapping tiles and filter land."""

    if not detections:
        return []

    lons_arr = np.array([d[0] for d in detections])
    lats_arr = np.array([d[1] for d in detections])
    confs_arr = np.array([d[2] for d in detections])

    # Filter detections on land using sea mask
    rows_px = ((bbox[3] - lats_arr) / (bbox[3] - bbox[1]) * vv_shape[0]).astype(int)
    cols_px = ((lons_arr - bbox[0]) / (bbox[2] - bbox[0]) * vv_shape[1]).astype(int)
    rows_px = np.clip(rows_px, 0, sea_mask.shape[0] - 1)
    cols_px = np.clip(cols_px, 0, sea_mask.shape[1] - 1)

    sea_filter = sea_mask[rows_px, cols_px]
    detections = [(lons_arr[i], lats_arr[i], confs_arr[i])
                  for i in range(len(detections)) if sea_filter[i]]

    print(f"After sea mask filter: {len(detections)}")

    # NMS - remove duplicates closer than threshold
    det_arr = np.array([(d[0], d[1]) for d in detections])
    conf_arr = np.array([d[2] for d in detections])
    keep = np.ones(len(det_arr), dtype=bool)

    for i in range(len(det_arr)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(det_arr)):
            if not keep[j]:
                continue
            dist = np.sqrt((det_arr[i, 0] - det_arr[j, 0])**2 +
                           (det_arr[i, 1] - det_arr[j, 1])**2)
            if dist < config.NMS_THRESHOLD_DEG:
                if conf_arr[i] >= conf_arr[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    detections_final = [detections[i] for i in range(len(detections)) if keep[i]]
    print(f"After deduplication: {len(detections_final)}")

    return detections_final


def save_geojson(detections, output_path):
    """Save detections as GeoJSON file."""

    features = []
    for lon, lat, conf in detections:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "confidence": round(conf, 4)
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"Saved {len(features)} detections to {output_path}")


def main(bbox, date):
    print(f"Starting pipeline for {date}, bbox: {bbox}")

    # Check bbox size
    check_bbox_size(bbox)

    # Load YOLO model
    print("Loading YOLO model...")
    model_path = hf_hub_download(
        repo_id=config.MODEL_REPO,
        filename=config.MODEL_FILE
    )
    model = YOLO(model_path)

    # 1. Download data
    print("\n[1/4] Downloading Sentinel-1 data...")
    vv_merged = search_and_download(bbox, date)

    # 2. Sea mask
    print("\n[2/4] Creating sea mask...")
    vv_clean = np.nan_to_num(vv_merged, nan=0.0)
    sea_mask = create_sea_mask(vv_clean, bbox)

    # 3. YOLO inference
    print("\n[3/4] Running YOLO inference...")
    detections = run_yolo(vv_clean, sea_mask, model, bbox)

    # 4. Deduplicate and export
    print("\n[4/4] Deduplicating and saving...")
    detections_final = deduplicate(detections, sea_mask, bbox, vv_clean.shape)

    output_path = os.path.join(config.OUTPUT_DIR, f"ships_{date}.geojson")
    save_geojson(detections_final, output_path)

    print(f"\nDone! Found {len(detections_final)} ships.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=config.DATE)
    parser.add_argument("--bbox", type=str, default=None)
    args = parser.parse_args()

    bbox = config.BBOX if args.bbox is None else [float(x) for x in args.bbox.split()]

    main(bbox, args.date)