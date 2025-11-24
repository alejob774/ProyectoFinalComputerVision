import argparse
from modulos.predict_video_to_excel import predict_video_to_excel
from modulos.visualizer import visualizer as generar_visualizacion
from modulos.conjunto import analizar_tiro_conjunto

def main():
    parser = argparse.ArgumentParser(description="Herramientas de analisis de tiro de baloncesto")
    parser.add_argument("--modo", type=str, required=True,
                        choices=["excel", "visualizer", "conjunto"],
                        help="Selecciona el modo a ejecutar")
    parser.add_argument("--video", type=str, required=True,
                        help="Ruta al video de entrada")

    args = parser.parse_args()

    if args.modo == "excel":
        predict_video_to_excel(args.video)

    elif args.modo == "visualizer":
        generar_visualizacion(args.video)

    elif args.modo == "conjunto":
        analizar_tiro_conjunto(args.video)

if __name__ == "__main__":
    main()
