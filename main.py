# import all other scripts
from data_load import data_load
from fmri_preprocessing import fmri_preprocessing
from spacial_transformations import spacial_transformations
from baseline_feature_extractor import baseline_feature_extractor
from motion_feature_extractor import motion_feature_extractor
from face_feature_extractor import face_feature_extractor
from body_feature_extractor import body_feature_extractor
from object_feature_extractor import object_feature_extractor
from scene_feature_extractor import scene_feature_extractor
from PCA import PCA
from modelling import train_predictor, evaluate


model = "Baseline"
# model = "Specialized"
transformations = False  # set to true for task 4

# load data in case it has not been downloaded locally
data_load()

# transform fmri data in case it has not been saved in a transformed format
fmri_preprocessing()

if transformations:
    spacial_transformations()

# create feature maps from videos in case they have not been created
if model == "Baseline":
    baseline_feature_extractor()
    motion_feature_extractor()
if model == "Specialized":
    face_feature_extractor()
    body_feature_extractor()
    object_feature_extractor()
    scene_feature_extractor()
    motion_feature_extractor()

data = PCA()

train_predictor(data)
evaluate()




