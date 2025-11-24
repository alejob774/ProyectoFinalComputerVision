import cv2
import math
import pandas as pd
from ultralytics import YOLO

# Clases según tu dataset
BALL_CLASS_IDS = [0, 2]    # basketball, sports ball
RIM_CLASS_IDS = [1]        # rim

REAL_RIM_WIDTH_METERS = 0.45  # 45 cm (ancho real del aro)


def predict_video_to_excel(video_path, model_path="best.pt", output_excel="resultado_tiro.xlsx"):

    model = YOLO(model_path, task="detect")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ No se pudo abrir el video.")
        return

    frame_count = 0
    trajectory = []
    rim_width_pixels = None

    # ------------------------------
    # Procesar cada frame
    # ------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, verbose=False)

        # ----- Buscar ARO -----
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in RIM_CLASS_IDS:
                    x1, y1, x2, y2 = box.xyxy[0]
                    width_px = float(x2 - x1)
                    if rim_width_pixels is None or width_px > rim_width_pixels:
                        rim_width_pixels = width_px

        # ----- Buscar BALÓN -----
        ball_found = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls in BALL_CLASS_IDS:
                    ball_found = True

                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)

                    trajectory.append((frame_count, cx, cy))

        if not ball_found:
            print(f"⚠ Frame {frame_count}: balón no detectado.")

    cap.release()

    # ------------------------------
    # Validaciones
    # ------------------------------
    if len(trajectory) < 2:
        print("❌ No se detectó trayectoria suficiente.")
        return

    if rim_width_pixels is None:
        print("❌ No se pudo detectar el aro para calibrar distancia.")
        return

    # ==============================
    # CALIBRACIÓN PIXEL → METRO
    # ==============================
    PIXELS_PER_METER = rim_width_pixels / REAL_RIM_WIDTH_METERS

    # ------------------------------
    # Punto MÁS ALTO (min Y)
    # ------------------------------
    highest_point = min(trajectory, key=lambda p: p[2])
    f_high, x_high, y_high = highest_point

    # ------------------------------
    # Punto FINAL del tiro
    # ------------------------------
    last_point = trajectory[-1]
    f_last, x_last, y_last = last_point

    # ------------------------------
    # ÁNGULO del tiro
    # ------------------------------
    dx = x_high - x_last
    dy = y_last - y_high   # invertir eje y de imagen

    angle_deg = math.degrees(math.atan2(dy, dx)) if dx != 0 else 90.0

    # ------------------------------
    # VELOCIDAD PROMEDIO EN X (m/s)
    # ------------------------------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    total_dx_pixels = x_last - trajectory[0][1]
    total_time = len(trajectory) / fps if fps > 0 else 1.0

    total_dx_meters = total_dx_pixels / PIXELS_PER_METER
    avg_velocity_x = abs(total_dx_meters / total_time)

    # ------------------------------
    # Guardar Excel (4 filas)
    # ------------------------------
    data = [
        ["Punto Más Alto", f_high, x_high, y_high, ""],
        ["Última Detección", f_last, x_last, y_last, ""],
        ["Ángulo del Tiro", "", "", "", angle_deg],
        ["Velocidad Promedio X (m/s)", "", "", "", avg_velocity_x]
    ]

    df = pd.DataFrame(data, columns=["Métrica", "Frame", "X", "Y", "Valor"])
    df.to_excel(output_excel, index=False)

    print(f"✔ Excel generado correctamente en: {output_excel}")
