import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
FONT = 'Arial'
FONT_SIZE = 8
LINEWIDTH = 0.5
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.transparent': True,
    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': FONT,
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    # line width
    'axes.linewidth': LINEWIDTH,
    'grid.linewidth': LINEWIDTH,
    'lines.linewidth': LINEWIDTH,
    'patch.linewidth': LINEWIDTH,
    'xtick.major.width': LINEWIDTH,
    'xtick.minor.width': LINEWIDTH,
    'ytick.major.width': LINEWIDTH,
    'ytick.minor.width': LINEWIDTH,
})

all_surp = []
all_freq = []
all_diss = []

stories = ['2_D', '4_D', '5_D', '6_D', '9_D']

for story in stories:

    # Surprisal 
    df_surp = pd.read_csv(
        "path_to_features/Surprisal0FF_St" + story + ".csv",
        sep=";"
    )
    df_surp["surpr_norm"] = (
        (df_surp["surprisal"] - df_surp["surprisal"].min()) /
        (df_surp["surprisal"].max() - df_surp["surprisal"].min())
    )
    df_surp.dropna(inplace=True)
    df_surp["story_id"] = story
    all_surp.append(df_surp)

    # Word frequency
    df_freq = pd.read_csv(
        "path_to_features/WordFreqOFF_St" + story + ".csv"
    )
    df_freq["Zipf_freq_norm"] = (
        (df_freq["Zipf_freq"] - df_freq["Zipf_freq"].min()) /
        (df_freq["Zipf_freq"].max() - df_freq["Zipf_freq"].min())
    )
    df_freq.dropna(inplace=True)
    df_freq["story_id"] = story
    all_freq.append(df_freq)

    # Semantic dissimilarity
    df_diss = pd.read_csv(
        "path_to_features/SemanticsDissOFF_St" + story + ".csv"
    )
    df_diss["semantic_dissimilarity_norm"] = (
        (df_diss["semantic_dissimilarity"] - df_diss["semantic_dissimilarity"].min()) /
        (df_diss["semantic_dissimilarity"].max() - df_diss["semantic_dissimilarity"].min())
    )
    df_diss["story_id"] = story
    all_diss.append(df_diss)

# Concatenate
df_surp_conc = pd.concat(all_surp, ignore_index=True)
df_freq_conc = pd.concat(all_freq, ignore_index=True)
df_diss_conc = pd.concat(all_diss, ignore_index=True)

# Final dataframe
df = pd.concat(
    [df_surp_conc,
     df_freq_conc["Zipf_freq"],
     df_diss_conc["semantic_dissimilarity"]],
    axis=1
)

df.drop(columns=["Unnamed: 0", "Column1"], errors="ignore", inplace=True)

print(df.shape)



# Normalizzazione tra 0 e 1 delle variabili numeriche
scaler = MinMaxScaler()

df_norm = df.copy()
df_norm[["surprisal", "semantic_dissimilarity", "Zipf_freq"]] = scaler.fit_transform(
    df[["surprisal", "semantic_dissimilarity", "Zipf_freq"]]
)

# PLOTS


fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# Surprisal vs Semantic Dissimilarity
sns.scatterplot(
    data=df_norm,
    x="surprisal",
    y="semantic_dissimilarity",
    hue="story_id",
    palette="BuPu",
    ax=axes[0],
    legend=False
)
axes[0].set_title("Surprisal vs Semantic Dissimilarity")
axes[0].set_xlabel("Surprisal")
axes[0].set_ylabel("Semantic Dissimilarity")

# Zipf Frequency vs Surprisal
sns.scatterplot(
    data=df_norm,
    x="Zipf_freq",
    y="surprisal",
    hue="story_id",
    palette="BuPu",
    ax=axes[1],
    legend=False
)
axes[1].set_title("Zipf Frequency vs Surprisal")
axes[1].set_xlabel("Zipf Frequency")
axes[1].set_ylabel("Surprisal")

# Zipf Frequency vs Semantic Dissimilarity
sns.scatterplot(
    data=df_norm,
    x="Zipf_freq",
    y="semantic_dissimilarity",
    hue="story_id",
    palette="BuPu",
    ax=axes[2],
    legend=False
)
axes[2].set_title("Zipf Frequency vs Semantic Dissimilarity")
axes[2].set_xlabel("Zipf Frequency")
axes[2].set_ylabel("Semantic Dissimilarity")

plt.tight_layout()
plt.show()