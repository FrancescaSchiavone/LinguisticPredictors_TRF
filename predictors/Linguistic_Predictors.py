import numpy as np
import pandas as pd

#parameters
FS = 100                  # target sampling rate (Hz)
AUDIO_FS = 44100          # original audio sampling rate (Hz)
DURATION_S = 3 * 60 + 43  # total duration of the stimulus (s)

CSV_FEATURES = "path_to_feature_file.csv"
XLSX_BOUNDS = "path_to_first_and_last_words.xlsx"

FEATURE_COLUMN = "feature_name"   # e.g., surprisal, entropy, frequency, etc.
ONSET_COLUMN = "BEGIN"

START_ROW = 0    # row index for first word
END_ROW = 1      # row index for last word

OUTPUT_FILE = "output_predictor.csv"

#load feature data 

df = pd.read_csv(CSV_FEATURES)     # if needed: sep=";"
onsets_audio = df[ONSET_COLUMN].to_numpy(dtype=float)
feature = df[FEATURE_COLUMN].to_numpy(dtype=float)

#load first_last word excel file 
bounds = pd.read_excel(XLSX_BOUNDS)
start_first = float(bounds.loc[START_ROW, "BEGIN"])
end_last = float(bounds.loc[END_ROW, "END"])

#empty predicotor
N = int(round(DURATION_S * FS))
predictor = np.zeros(N, dtype=float)

# Convert onsets from AUDIO_FS to FS (0-based)
idx = np.floor(onsets_audio / AUDIO_FS * FS).astype(int)
valid = (idx >= 0) & (idx < N)

predictor[idx[valid]] = feature[valid]

cut_begin = int(np.floor(start_first / AUDIO_FS * FS))
cut_end = int(np.ceil(end_last / AUDIO_FS * FS))

cut_begin = max(0, cut_begin)
cut_end = min(N, cut_end)

predictor_cut = predictor[cut_begin:cut_end]

#save file
np.savetxt(OUTPUT_FILE, predictor_cut, delimiter=",")


