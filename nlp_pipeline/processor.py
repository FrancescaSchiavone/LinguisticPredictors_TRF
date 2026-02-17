import os
import logging
import string
import pandas as pd
import stanza

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

stanza.download("it")
nlp = stanza.Pipeline("it", processors="tokenize, mwt, pos, lemma, ", use_gpu=False)

def process_text_file(filepath, output_dir): 
    """
    Processes a single text file using the Stanza NLP pipeline and extracts linguistic features.

    For each input file, a dedicated subfolder is created (named after the file, e.g. '01_03') 
    where all outputs are saved.

    Parameters:
        filepath (str): Path to the input .txt file.
        output_dir (str): Root directory where output subfolders will be created.

    Outputs:
        - A CSV file containing token-level linguistic features:
          sentence ID, token, lemma, PoS,  cleaned token/lemma, and SUBTLEX-IT Zipf frequency.

        

    Notes:
        - Output filenames are automatically derived from the input filename.
        - Constituency parsing is included if available in the sentence object.
        - AoA and frequency values are mapped from external normative datasets.
        - Logging messages indicate where each output file is saved.

    """
    logging.info(f"Processing file: {filepath}")

    with open(filepath, "r", encoding="utf-8") as infile:
        text = infile.read()

    doc = nlp(text)
    sentence_ids, tokens, PoS, lemma, clean_tokens, clean_lemmas, raw_tokens = [], [], [], [], [], [], []
    for sent_id, sentence in enumerate(doc.sentences):
        for token in sentence.tokens:
            word = token.words[0]
            if word.pos != "PUNCT":
                sentence_ids.append(sent_id)
                raw_token = token.text
                raw_lemma = word.lemma        
                clean_token = raw_token.lower().translate(str.maketrans("","", string.punctuation)) 
                clean_lemma = raw_lemma.translate(str.maketrans("","",string.punctuation))
                clean_tokens.append(clean_token)
                clean_lemmas.append(clean_lemma)
                raw_tokens.append(raw_token)
                PoS.append(word.pos)
              

    subtlex_df = pd.read_excel('PATH')
    subtlex_df["zipf"] = subtlex_df["zipf"]
    freq_dict = subtlex_df.set_index("wordform")["zipf"].to_dict()

    df = pd.DataFrame({
        "sentence_ids": sentence_ids,
        "tokens_no_punct": clean_tokens, 
        "lemma_no_punct": clean_lemmas,
        "PoS": PoS
    })
    

    function_pos = {"ADP", "AUX", "CCONJ", "SCONJ", "DET", "PRON", "PART", "INTJ", "ADV"}
    content_pos = {"NOUN", "VERB", "ADJ", "PROPN"}

    df["type_of_words"] = df["PoS"].apply(lambda x: "function" if x in function_pos else ("content" if x in content_pos else 'NaN'))
    df["Zipf_freq"] = df["tokens_no_punct"].map(freq_dict)

    df["token_id"] = range(1, len(df)+1)
    df = df[["token_id", "sentence_ids", "tokens_no_punct", "lemma_no_punct", "PoS", "depparse", "head", "constituency", "AoA(m+sd)", "Zipf_freq", "type_of_words"]]


    name_base = os.path.splitext(os.path.basename(filepath))[0]
    file_output_dir = os.path.join (output_dir, name_base)
    os.makedirs(file_output_dir, exist_ok=True)
    csv_path = os.path.join(file_output_dir, f"{name_base}.csv")
    df.to_csv(csv_path, sep=";",  decimal=",", index=False, encoding="utf-8-sig")
    logging.info(f"Saved CSV: {csv_path}")
    






    

