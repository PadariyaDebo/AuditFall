"""
Configuration constants for the AuditFall pipeline:
"""

import os


BASE_DIR = os.environ.get("SISFALL_DIR", "/content/SisFall_dataset")

# Activity code prefixes in the SisFall filenames, following the original
# dataset definition (Sucerquia et al., Sensors 2017, 17, 198):
# F01-F15 = falls, D01-D19 = activities of daily living (ADLs)
FALL_PREFIX = "F"
ADL_PREFIX = ("D",)

RANDOM_STATE = 0
TEST_SIZE = 0.25
CV_FOLDS = 5

# 9 sensor channels x 8 stats + 4 acceleration-magnitude features = 76
SENSOR_CHANNELS = [
    "ADXL345_x", "ADXL345_y", "ADXL345_z",
    "ITG3200_x", "ITG3200_y", "ITG3200_z",
    "MMA8451Q_x", "MMA8451Q_y", "MMA8451Q_z",
]
STAT_NAMES = ["Mean", "Median", "Std", "Max", "Min", "Var", "Skewness", "Kurtosis"]
MAG_FEATURE_NAMES = ["Mag_Mean", "Mag_Std", "Mag_Max", "Mag_Min"]

FEATURE_NAMES = [f"{ch}_{st}" for ch in SENSOR_CHANNELS for st in STAT_NAMES] + MAG_FEATURE_NAMES

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
