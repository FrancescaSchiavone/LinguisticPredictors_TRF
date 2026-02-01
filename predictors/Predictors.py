import pandas as pd
import numpy as np

# Parameters
fs = 100  # final Hz for the predictor
durata_totale = 3 * 60 + 43  # 223 seconds 

csv_file= r'Linguistic Features\Surprisal0FF_St9_D.csv' 
csv_file_first_last_word =  "first_and_last_words_stories_D.xlsx"

# Import CSV Surprisal
T = pd.read_csv(csv_file) #if error: sep=";"
onset_44100 = T['BEGIN'].values
#onset_44100 = T['BEGIN'].str.replace(",", ".").astype(float).to_numpy() # samples at 44100 Hz #in case of error
surprisal = T["surprisal"].values 


surprisal_min = np.min(surprisal)
surprisal_max = np.max(surprisal)

surprisal = (surprisal - surprisal_min) / (surprisal_max - surprisal_min)


# Import first and last word info
T2 = pd.read_excel(csv_file_first_last_word)
start_first_word = T2.loc[13, "BEGIN"]    
end_last_word = T2.loc[14, "END"]        

# Empty predictor
N = round(durata_totale * fs)
pred = np.zeros(N)

# Convert onset from 44100 Hz to 100 Hz
idx = np.floor(onset_44100 / 44100 * fs).astype(int)

cut_beginning = int(np.floor(start_first_word / 44100 * fs))
cut_end = int(np.ceil(end_last_word / 44100 * fs))
cut_beginning = max(1, cut_beginning)
cut_end = min(N, cut_end)

# Put surprisal values at each onset
for i in range(len(idx)):
    if 1 <= idx[i] <= N:
        pred[idx[i] - 1] = surprisal[i]
        print('done')


pred_cut= pred[cut_beginning - 1 : cut_end]

np.savetxt("surp9_01.csv", pred_cut, delimiter=",") #🫖



#WORDONSET GUASS
import numpy as np
import pandas as pd

fs = 100                                # final Hz
durata_totale = 3 * 60 + 43             # 223 seconds

csv_file = r'Linguistic Features\SemanticsDissOFF_St6_D.csv'
csv_file_first_last_word = "first_and_last_words_stories_D.xlsx"

sigma = 1.0      # ~10 ms (a 100 Hz)
radius = 3       # ±30 ms
x = np.arange(-radius, radius + 1)
gauss = np.exp(-(x**2) / (2 * sigma**2))
gauss = gauss / gauss.max()             

T = pd.read_csv(csv_file)                # if error: sep=";"
onset_44100 = T['BEGIN'].values.astype(float)

T2 = pd.read_excel(csv_file_first_last_word)
start_first_word = T2.loc[1, "BEGIN"]
end_last_word = T2.loc[2, "END"]


N = round(durata_totale * fs)
pred = np.zeros(N)


idx = np.floor(onset_44100 / 44100 * fs).astype(int)

cut_beginning = int(np.floor(start_first_word / 44100 * fs))
cut_end = int(np.ceil(end_last_word / 44100 * fs))
cut_beginning = max(0, cut_beginning)
cut_end = min(N, cut_end)


for i in idx:
    center = i
    for dx, g in zip(x, gauss):
        pos = center + dx
        if 0 <= pos < N:
            pred[pos] += g   # ampiezza = 1


pred_cut = pred[cut_beginning : cut_end]


np.savetxt("wo_gauss2.csv", pred_cut, delimiter=",")


