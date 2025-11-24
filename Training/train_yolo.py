#!/usr/bin/env python3
"""
finetune_yolo_simple.py

Fine-tuning de YOLOv8 usando únicamente la ruta al ZIP del dataset.
El script descomprime el ZIP, busca automáticamente el archivo .yaml
y entrena el modelo con early stopping.
"""

# Ruta al dataset en formato ZIP
ZIP_DATASET_PATH = "C:\\Users\\Alejandro\\Downloads\\Basket_Project\\Datasets\\Basketball.v3i.yolov8.zip"

# Modelo base YOLOv8
BASE_MODEL = "yolov8n.pt"

import os
import zipfile
from ultralytics import YOLO

def main():
    # 1. Descomprimir dataset
    extract_dir = "dataset"
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(ZIP_DATASET_PATH, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✅ Dataset descomprimido en '{extract_dir}/'")

    # 2. Buscar archivo .yaml dentro de la carpeta principal
    data_yaml = None
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.endswith(".yaml") or f.endswith(".yml"):
                data_yaml = os.path.join(root, f)
                break
        if data_yaml:
            break

    if not data_yaml:
        raise FileNotFoundError("❌ No se encontró ningún archivo .yaml en el dataset descomprimido.")

    print(f"📂 Archivo de configuración encontrado: {data_yaml}")

    # 3. Cargar modelo base
    model = YOLO(BASE_MODEL)

    # 4. Entrenar con early stopping
    results = model.train(
        data=data_yaml,
        epochs=300,      # máximo de épocas
        patience=25,      # early stopping
        imgsz=640,
        batch=16,
        device=0,         # usa GPU si está disponible
        verbose=True
    )

    # 5. Resumen final
    print("\n📊 Entrenamiento finalizado.")
    print(f"Mejor época alcanzada: {results.best_epoch}")
    print(f"Mejor mAP50-95: {results.metrics['metrics/mAP50-95(B)']:.4f}")
    print(f"Mejor precisión: {results.metrics['metrics/precision(B)']:.4f}")
    print(f"Mejor recall: {results.metrics['metrics/recall(B)']:.4f}")

if __name__ == "__main__":
    main()
