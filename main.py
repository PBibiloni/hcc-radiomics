import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from lib import features
from lib.dataset import Radcttace

ROOT_PATH = Path(__file__).parent / 'data' / 'RadCTTACEomics_v2024.05'
OUTPUT_PATH = Path(__file__).parent / 'output' / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
DELETE_PREVIOUS_OUTPUT_DIRS = True
SAVE_MEDIAN_AXIAL_SLICES = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


if DELETE_PREVIOUS_OUTPUT_DIRS:
    for output_dir in OUTPUT_PATH.parent.iterdir():
        if output_dir.is_dir():
            shutil.rmtree(output_dir)

# Initialize dataset
dataset = Radcttace(path=ROOT_PATH)
dataset_train, _ = dataset.split(ratio=0.10)

# Extract features
features_per_patient = {}
for idx, patient in enumerate(dataset_train.patients):
    log.info(f'Processing patient {patient.name} (Progress {(idx+1)/len(dataset_train.patients):%})')
    acq = patient.main_acquisition()
    ct = acq.ct()
    seg_tumor = acq.segmentation_tumor()
    seg_liver = acq.segmentation_liver()

    if SAVE_MEDIAN_AXIAL_SLICES:
        fig, ax = plt.subplots()
        acq.plot_slice(ax, vmin=0, vmax=400)
        os.makedirs(OUTPUT_PATH / 'MedianAxialSlices', exist_ok=True)
        fig.savefig(OUTPUT_PATH / 'MedianAxialSlices' / f'{patient.name}.png')
        # fig.show()
        plt.close(fig)

    # Generate derived segmentation masks
    segmentations = {
        'Tumor': seg_tumor,
        # 'Liver': seg_liver,
        # 'TumorPerimeterInner': inner_perimeter_mask(seg_tumor, width_mm=10),
        # 'TumorPrimeterOuter': outer_perimeter_mask(seg_tumor, seg_liver, width_mm=10),
        # 'TumorCore': tumor_core_mask(seg_tumor, radius_mm=10)
    }

    # Compute features
    all_features = {}
    for seg_name, seg in segmentations.items():
        for feature_name, feature_value in features.dummy_features(ct=ct, segmentation=seg).items():
            all_features[f'{seg_name}_{feature_name}'] = feature_value
    features_per_patient[patient.name] = all_features

# Save all features
df_features = pd.DataFrame(features_per_patient).T
os.makedirs(OUTPUT_PATH, exist_ok=True)
df_features.to_csv(OUTPUT_PATH / 'features.csv')

# Load objective function
df_objective = pd.Series(
    [patient.objective_function() for patient in dataset_train.patients],
    index=[patient.name for patient in dataset_train.patients],
    name='Recist',
)
df_objective.to_csv(OUTPUT_PATH / 'objective.csv')

# Train classifier model
model = RandomForestClassifier()
model.fit(df_features, df_objective)
predictions = model.predict(df_features)
df_predictions = pd.Series(predictions, index=df_features.index, name='Predictions')
df_objective.to_csv(OUTPUT_PATH / 'prediction.csv')

# Assess correctness of predicitions
accuracy = (df_predictions == df_objective.values).mean()
print(f'Accuracy: {accuracy:%}')
