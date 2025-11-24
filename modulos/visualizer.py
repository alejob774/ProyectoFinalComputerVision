import cv2
import math
from ultralytics import YOLO

BALL_CLASS_IDS = [0, 2]
RIM_CLASS_IDS = [1]

REAL_RIM_WIDTH_METERS = 0.45


def draw_circle(img, x, y, color, radius=7, thickness=-1):
    cv2.circle(img, (int(x), int(y)), radius, color, thickness)


def visualizer(video_path, model_path="best.pt", output_image="visualizacion_tiro.png"):

    model = YOLO(model_path, task="detect")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ No se pudo abrir el video.")
        return

    trajectory = []
    rim_width_pixels = None
    last_frame_with_ball = None

    frame_count = 0

    # =====================================================
    # 1. Recorrer video para obtener trayectoria y aro
    # =====================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, verbose=False)

        # Detectar aro
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in RIM_CLASS_IDS:
                    x1, y1, x2, y2 = box.xyxy[0]
                    width_px = float(x2 - x1)

                    if rim_width_pixels is None or width_px > rim_width_pixels:
                        rim_width_pixels = width_px

        # Detectar balón
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

                    last_frame_with_ball = frame.copy()

        if not ball_found:
            print(f"⚠ Frame {frame_count}: balón no detectado.")

    cap.release()

    # ========================
    # Validaciones
    # ========================
    if len(trajectory) < 2:
        print("❌ No se detectó trayectoria suficiente.")
        return

    if last_frame_with_ball is None:
        print("❌ No se pudo obtener imagen final del balón.")
        return

    # ========================
    # Selección de puntos
    # ========================
    highest_point = min(trajectory, key=lambda p: p[2])
    last_point = trajectory[-1]
    first_point = trajectory[0]

    # ========================
    # DIBUJAR RESULTADO
    # ========================
    img = last_frame_with_ball.copy()

    # Dibujar trayectoria
    for i in range(1, len(trajectory)):
        _, x1, y1 = trajectory[i - 1]
        _, x2, y2 = trajectory[i]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Dibujar puntos individuales
    for _, x, y in trajectory:
        draw_circle(img, x, y, (255, 255, 0), radius=4)

    # PUNTO DE LANZAMIENTO
    draw_circle(img, first_point[1], first_point[2], (0, 0, 255), radius=8)

    # PUNTO MÁS ALTO
    draw_circle(img, highest_point[1], highest_point[2], (0, 255, 255), radius=10)

    # ÚLTIMA DETECCIÓN
    draw_circle(img, last_point[1], last_point[2], (255, 0, 0), radius=10)

    # GUARDAR IMAGEN
    cv2.imwrite(output_image, img)
    print(f"✔ Imagen generada correctamente: {output_image}")
