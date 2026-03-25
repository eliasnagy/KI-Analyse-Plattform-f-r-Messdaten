"""
Zentrale Konfigurationsdatei - lädt alle Einstellungen aus .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Lade .env Datei aus dem Projektverzeichnis
ENV_PATH = Path(__file__).parent / '.env'
load_dotenv(ENV_PATH)


class Config:
    """Zentrale Konfigurationsklasse"""
    
    # ==========================================
    # FOLDER PFADE
    # ==========================================
    BASE_TRAINING_DIR = os.getenv('BASE_TRAINING_DIR', './trainings_daten')
    INPUT_FOLDERS_STR = os.getenv('INPUT_FOLDERS', 'data_files/c1')  # Komma-getrennte Liste
    INPUT_FOLDERS = [f.strip() for f in INPUT_FOLDERS_STR.split(',')]
    WEAR_FILES_FOLDER = os.getenv('WEAR_FILES_FOLDER', 'wear_files')
    NUMPY_FILES_FOLDER = os.getenv('NUMPY_FILES_FOLDER', 'numpy_files')
    OUTPUT_FILES_FOLDER = os.getenv('OUTPUT_FILES_FOLDER', 'output_files')
    
    # Lazy-Evaluation für volle Pfade
    @classmethod
    def _setup_paths(cls):
        cls.TRAIN_INPUT_PATHS = [os.path.join(cls.BASE_TRAINING_DIR, folder) for folder in cls.INPUT_FOLDERS]
        cls.WEAR_FILES_PATH = os.path.join(cls.BASE_TRAINING_DIR, cls.WEAR_FILES_FOLDER)
    
    # Default volle Pfade
    TRAIN_INPUT_PATHS = None  # Wird in init_paths gesetzt
    WEAR_FILES_PATH = None
    
    # ==========================================
    # RANDOM FOREST PARAMETER
    # ==========================================
    RF_N_ESTIMATORS = int(os.getenv('RF_N_ESTIMATORS', '100'))
    RF_MAX_DEPTH = None if os.getenv('RF_MAX_DEPTH', 'None') == 'None' else int(os.getenv('RF_MAX_DEPTH'))
    
    # ==========================================
    # MLP PARAMETER
    # ==========================================
    MLP_HIDDEN_LAYERS = os.getenv('MLP_HIDDEN_LAYERS', '100,50')
    MLP_MAX_ITER = int(os.getenv('MLP_MAX_ITER', '1000'))
    MLP_ACTIVATION = os.getenv('MLP_ACTIVATION', 'relu')
    MLP_SOLVER = os.getenv('MLP_SOLVER', 'adam')
    
    # ==========================================
    # DATEN PARAMETER
    # ==========================================
    TEST_SPLIT_RATIO = float(os.getenv('TEST_SPLIT_RATIO', '0.2'))
    RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))
    
    # ==========================================
    # FEATURE EXTRACTION
    # ==========================================
    COLUMN_NAMES = os.getenv('COLUMN_NAMES', 'Force_X,Force_Y,Force_Z,Vibration_X,Vibration_Y,Vibration_Z,AE_RMS').split(',')
    
    @staticmethod
    def print_config():
        """Gebe alle Konfigurationsparameter aus"""
        Config._setup_paths()  # Setup Pfade falls nötig
        
        print("\n" + "="*70)
        print("KONFIGURATION GELADEN")
        print("="*70)
        print(f"Base Training Dir:    {Config.BASE_TRAINING_DIR}")
        print(f"Input Folders ({len(Config.INPUT_FOLDERS)}):") 
        for folder in Config.INPUT_FOLDERS:
            path = os.path.join(Config.BASE_TRAINING_DIR, folder)
            print(f"  - {folder}")
        print(f"Wear Files:           {Config.WEAR_FILES_PATH if Config.WEAR_FILES_PATH else os.path.join(Config.BASE_TRAINING_DIR, Config.WEAR_FILES_FOLDER)}")
        print(f"  (Wear-Datei wird automatisch erkannt!)")
        print(f"NumPy Cache:          {Config.NUMPY_FILES_FOLDER}")
        print(f"Output Modelle:       {Config.OUTPUT_FILES_FOLDER}")
        print(f"\nRandom Forest:")
        print(f"  Estimators: {Config.RF_N_ESTIMATORS}, Max Depth: {Config.RF_MAX_DEPTH}")
        print(f"\nMLP:")
        print(f"  Hidden Layers: {Config.MLP_HIDDEN_LAYERS}, Max Iter: {Config.MLP_MAX_ITER}")
        print(f"  Activation: {Config.MLP_ACTIVATION}, Solver: {Config.MLP_SOLVER}")
        print(f"\nDaten:")
        print(f"  Test Split: {Config.TEST_SPLIT_RATIO}, Random State: {Config.RANDOM_STATE}")
        print("="*70 + "\n")
