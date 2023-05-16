from utils import clean, load_pickle, normalize, remove_stop, replace_urls, save_pickle, tokenize, prepare

pipeline = [str.lower, tokenize, remove_stop]

def clean_all(df, filename):
    df['clean_title'] = df['title'].progress_map(clean)
    df['clean_study_title'] = df['study_title'].progress_map(clean)
    df['clean_abstract'] = df['study_abstract'].progress_map(clean)

    df['clean_title'] = df['clean_title'].progress_map(replace_urls)
    df['clean_study_title'] = df['clean_study_title'].progress_map(replace_urls)
    df['clean_abstract'] = df['clean_abstract'].progress_map(replace_urls)
    df['clean_title'] = df['clean_title'].progress_map(normalize)
    df['clean_study_title'] = df['clean_study_title'].progress_map(normalize)
    df['clean_abstract'] = df['clean_abstract'].progress_map(normalize)
    df['clean_text'] = df['clean_study_title'] + ". " + df["clean_abstract"]
    
    df['title_tokens'] = df['clean_title'].progress_apply(prepare, pipeline=pipeline)
    df['study_title_tokens'] = df['clean_study_title'].progress_apply(prepare, pipeline=pipeline)
    df['abstract_tokens'] = df['clean_abstract'].progress_apply(prepare, pipeline=pipeline)
    
    df['text_tokens'] = df['study_title_tokens'] + df["abstract_tokens"]
    save_pickle(filename, df)

df = load_pickle("./data/processed/df.pickle")
clean_all(df, "./data/processed/df_tokenized.pickle")