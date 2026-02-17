# Linguistic Predictors in Continuous Naturalistic Speech: An EEG-Based Analysis

> This repository contains the code used in the thesis.  
> It is not intended as a fully plug-and-play pipeline: users may need to adapt data paths and folder structures to their own setup.

## Repository Structure

The project is organized into the following folders:

- **`nlp_pipeline`**  
  Contains the code used to compute the linguistic features employed in the thesis:
  - word frequency (see `processor.py`)
  - surprisal (see `surprisal.py`) 
  - semantic dissimilarity  (see `semantic_dissimilarity.py`)

- **`predictors`**  
  Contains the code used to generate weighted predictors from the linguistic features.

- **`mTRF`**  
  Contains the code related to temporal response function (mTRF) modeling (mTRF.ipynb)

- **`Analysis`**  
  Contains scripts for statistical analysis and visualization of results:
  - correlation within linguistic features (see `correlation.py`)
  - model comparisons (full vs. reduced) (see `model_comparison.ipynb`)
  - N400-like effect of TRF (`n400-style.ipynb`)

## References

- Amenta, S., Mandera, P., Keuleers, E., Brysbaert, M., & Crepaldi, D. (2025, July 7).  
  **SUBTLEX-IT: Word frequency estimates for Italian based on movie subtitles**.  
  Retrieved from https://osf.io/zg7sc

- Bird, S., Loper, E., & Klein, E. (2009).  
  **Natural Language Processing with Python**. O'Reilly Media Inc.

- Parisi, L., Francia, S., & Magnani, P. (2020).  
  **UmBERTo: An Italian Language Model Trained with Whole Word Masking**.  
  https://huggingface.co/Musixmatch/umberto-commoncrawl-cased-v1

- Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020).  
  **Stanza: A Python Natural Language Processing Toolkit for Many Human Languages**.  
  ACL System Demonstrations.

- de Vries, W., & Nissim, M. (2021).  
  **As Good as New: How to Successfully Recycle English GPT-2 to Make Models for Other Languages**.  
  Findings of ACL-IJCNLP 2021.  
  https://huggingface.co/GroNLP/gpt2-small-italian-embeddings

- Brodbeck, C., Das, P., Gillis, M., Kulasingham, J. P., Bhattasali, S., Gaston, P., Resnik, P., & Simon, J. Z. (2023).  
  **Eelbrain: A Python Toolkit for Time-Continuous Analysis with Temporal Response Functions**.  
  *eLife, 12.*
