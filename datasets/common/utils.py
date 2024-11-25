import sys

def progress_bar(total, current, long=50):
    porcentaje = (current / total) * 100
    bloques = int((current / total) * long)
    barra = f"[{'#' * bloques}{'.' * (50 - bloques)}] {porcentaje:.2f}%"
    
    # Imprime la barra de progreso en la misma línea
    sys.stdout.write(f"\r{barra}")
    sys.stdout.flush()

    if current == total:
        print()