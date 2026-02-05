import numpy as np
import pandas as pd
import pickle

#parameters
FS = 100                      # Hz
TRIAL_SEC = 60                # seconds per trial
TRIAL_LEN = TRIAL_SEC * FS    # samples per trial
NUM_TRIALS = 15
TRIALS_PER_STORY = 3

PRED_FILES = [
    "path_to_predictor_1.csv",
    "path_to_predictor_2.csv",
    "path_to_predictor_3.csv",
    "path_to_predictor_4.csv",
    "path_to_predictor_5.csv",
]

OUT_PKL = "trials_SUBJECTID_PREDICTOR.pkl"

max_story_len = TRIALS_PER_STORY * TRIAL_LEN
chunks = []

for path in PRED_FILES:
    # works whether the CSV has header or not (common when saved via np.savetxt)
    x = pd.read_csv(path, header=None).to_numpy().ravel()
    chunks.append(x[:max_story_len])

pred_total = np.concatenate(chunks) if chunks else np.array([], dtype=float)

#creation trials
total_needed = NUM_TRIALS * TRIAL_LEN
padded = np.zeros(total_needed, dtype=float)
n_copy = min(len(pred_total), total_needed)
padded[:n_copy] = pred_total[:n_copy]

trials_array = padded.reshape(NUM_TRIALS, TRIAL_LEN)  # shape: (NUM_TRIALS, TRIAL_LEN)

#save pickle
with open(OUT_PKL, "wb") as f:
    pickle.dump({"trials": trials_array, "fs": FS}, f)
