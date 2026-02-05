import numpy as np
import pandas as pd

#parameters
FS = 100
AUDIO_FS = 44100
DURATION_S = 3 * 60 + 43

CSV_ONSETS = "path_to_onset_file.csv"
XLSX_BOUNDS = "path_to_first_and_last_words.xlsx"

ONSET_COLUMN = "BEGIN"
START_ROW = 0
END_ROW = 1

SIGMA = 1.0     # in bins (1 bin = 10 ms at 100 Hz)
RADIUS = 3      # ± bins around onset

OUTPUT_FILE = "output_wordonset_gauss.csv"

#Gaussian Kernel
x = np.arange(-RADIUS, RADIUS + 1)
gauss = np.exp(-(x**2) / (2 * SIGMA**2))
gauss /= gauss.max()

#load data
df = pd.read_csv(CSV_ONSETS)       # if needed: sep=";"
onsets_audio = df[ONSET_COLUMN].to_numpy(dtype=float)

bounds = pd.read_excel(XLSX_BOUNDS)
start_first = float(bounds.loc[START_ROW, "BEGIN"])
end_last = float(bounds.loc[END_ROW, "END"])

#buil predictor
N = int(round(DURATION_S * FS))
predictor = np.zeros(N, dtype=float)

idx = np.floor(onsets_audio / AUDIO_FS * FS).astype(int)

for center in idx:
    for dx, g in zip(x, gauss):
        pos = center + dx
        if 0 <= pos < N:
            predictor[pos] += g


cut_begin = int(np.floor(start_first / AUDIO_FS * FS))
cut_end = int(np.ceil(end_last / AUDIO_FS * FS))

cut_begin = max(0, cut_begin)
cut_end = min(N, cut_end)

predictor_cut = predictor[cut_begin:cut_end]

#save predictor
np.savetxt(OUTPUT_FILE, predictor_cut, delimiter=",")
