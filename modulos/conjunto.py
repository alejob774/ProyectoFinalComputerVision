import cv2
import math
from ultralytics import YOLO
import numpy as np

# IDs correctos segun tu modelo
BALL_CLASS_IDS = [0, 2]  # basketball + sports ball
RIM_CLASS_IDS = [1]      # rim

REAL_RIM_WIDTH_METERS = 0.45  # 45 cm


def draw_table_on_canvas(rows, width):
    """
    Crea una imagen tabla en fondo blanco.
    rows: lista de listas con 5 columnas
    width: ancho del canvas (igual al ancho de la imagen original)
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    col_width = width // 5
    row_height = 40

    table_height = row_height * (len(rows) + 1)

    table_img = 255 * np.ones((table_height, width, 3), dtype=np.uint8)

    # Dibujar líneas horizontales
    for i in range(len(rows) + 2):
        y = i * row_height
        cv2.line(table_img, (0, y), (width, y), (0, 0, 0), 2)

    # Dibujar líneas verticales
    for j in range(6):
        x = j * col_width
        cv2.line(table_img, (x, 0), (x, table_height), (0, 0, 0), 2)

    # Header
    header = ["Metrica", "Frame", "X", "Y", "Valor"]

    for col_idx, text in enumerate(header):
        text_x = col_idx * col_width + 10
        text_y = int(row_height * 0.7)
        cv2.putText(table_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    # Filas
    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            text_x = col_idx * col_width + 10
            text_y = (row_idx + 1) * row_height + int(row_height * 0.7)
            cv2.putText(table_img, str(cell), (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return table_img



def analizar_tiro_conjunto(video_path, output_image="visualizacion_con_tabla.png"):
    import numpy as np

    model = YOLO("best.pt", task="detect")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("No se pudo abrir video.")
        return

    frame_count = 0
    trajectory = []
    rim_width_pixels = None

    last_frame_img = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, verbose=False)

        # Detectar aro para calibracion
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in RIM_CLASS_IDS:
                    x1, y1, x2, y2 = box.xyxy[0]
                    w = float(x2 - x1)
                    if rim_width_pixels is None or w > rim_width_pixels:
                        rim_width_pixels = w

        # Detectar balon
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in BALL_CLASS_IDS:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)

                    trajectory.append((frame_count, cx, cy))
                    last_frame_img = frame.copy()

    cap.release()

    # Validaciones
    if len(trajectory) < 2:
        print("No se obtuvo trayectoria suficiente.")
        return

    if rim_width_pixels is None:
        print("No se detecto aro para calibrar.")
        return

    # ================================
    # CALIBRACION
    # ================================
    PIXELS_PER_METER = rim_width_pixels / REAL_RIM_WIDTH_METERS

    # ================================
    # METRICAS
    # ================================
    highest = min(trajectory, key=lambda t: t[2])
    last = trajectory[-1]

    frame_h, x_h, y_h = highest
    frame_l, x_l, y_l = last

    # ÁNGULO DEL TIRO (corregido)
    dx = x_h - x_l
    dy = y_l - y_h  # corrección por sistema de coordenadas de imagen
    angle_deg = math.degrees(math.atan2(dy, dx)) if dx != 0 else 90.0

    # Velocidad X (valor absoluto, en metros/s)
    dx_pix_total = x_l - trajectory[0][1]
    dx_meters = dx_pix_total / PIXELS_PER_METER

    cap2 = cv2.VideoCapture(video_path)
    fps = cap2.get(cv2.CAP_PROP_FPS)
    cap2.release()

    total_time = len(trajectory) / fps if fps > 0 else 1
    vel_x = abs(dx_meters / total_time)

    # ================================
    # VISUALIZACION
    # ================================
    final_img = last_frame_img.copy()

    # Dibujar trayectoria
    for i in range(1, len(trajectory)):
        (f1, x1, y1) = trajectory[i - 1]
        (f2, x2, y2) = trajectory[i]
        cv2.line(final_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Puntos
    cv2.circle(final_img, (int(x_h), int(y_h)), 8, (255, 0, 0), -1)
    cv2.circle(final_img, (int(x_l), int(y_l)), 8, (0, 0, 255), -1)

    # ================================
    # TABLA DE DATOS (debajo de la imagen)
    # ================================
    rows = [
        ["Punto Mas Alto", frame_h, round(x_h, 2), round(y_h, 2), ""],
        ["Ultima Deteccion", frame_l, round(x_l, 2), round(y_l, 2), ""],
        ["Angulo del Tiro", "", "", "", round(angle_deg, 2)],
        ["Velocidad X (m/s)", "", "", "", round(vel_x, 3)],
    ]

    table = draw_table_on_canvas(rows, final_img.shape[1])

    # Combinar imagen + tabla verticalmente
    combined = np.vstack((final_img, table))

    cv2.imwrite(output_image, combined)
    print(f"✔ Imagen con tabla generada: {output_image}")
