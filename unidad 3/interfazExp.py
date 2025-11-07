from pyswip import Prolog

prolog = Prolog()

prolog_file = "diagnosticoResp.pl"

try:
    prolog.consult(prolog_file)
    print(f"Archivo {prolog_file} cargado exitosamente.")
except Exception as e:
    print(f"ERROR: No se pudo cargar el archivo {prolog_file}.")
    print(f"Detalles del error: {e}")
    exit()

print("\n--- INICIANDO SISTEMA EXPERTO DE DIAGNÓSTICO ---\n")
try:
    list(prolog.query("iniciar_diagnostico"))
except Exception as e:
    print(f"\nOcurrió un error durante la ejecución del diagnóstico: {e}")

print("\n--- Sesión de diagnóstico finalizada ---")
