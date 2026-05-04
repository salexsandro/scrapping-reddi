import pandas as pd
import re
import emoji
import spacy
import time
import os
import gc

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tok2vec", "attribute_ruler"])

slang_dict = {
    "tbh": "to be honest", "imo": "in my opinion", "gf": "girlfriend",
    "bf": "boyfriend", "op": "original poster", "idk": "i dont know"
}

def clean_social_text(text):
    if not isinstance(text, str): return ""
    text = emoji.demojize(text)
    text = re.sub(r'r/\w+|u/\w+', '', text)
    text = re.sub(r'http\S+', '[URL]', text)
    text = re.sub(r'[^a-zA-Z0-9\s:_]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    words = [slang_dict.get(w, w) for w in text.split()]
    return " ".join(words)

def processar_ambiente_controlado(arquivo_entrada, arquivo_saida):
    inicio_total = time.time()
    
    if os.path.exists(arquivo_saida):
        os.remove(arquivo_saida)
        
    tamanho_chunk = 10000 
    print(f"Iniciando modo de segurança (Chunks: {tamanho_chunk}, Single-Thread)...")
    
    chunks = pd.read_csv(arquivo_entrada, chunksize=tamanho_chunk)
    
    for i, chunk in enumerate(chunks):
        inicio_chunk = time.time()
        
        chunk.dropna(subset=['text'], inplace=True)
        chunk.drop_duplicates(subset=['comment_id'], inplace=True)
        chunk = chunk[chunk['text'].str.len() > 10].copy() 
        
        chunk['cleaned_text'] = chunk['text'].apply(clean_social_text)
        chunk = chunk[chunk['cleaned_text'].str.strip() != '']
        
        textos_limpos = chunk['cleaned_text'].tolist()
        lemmatized_results = []
        
        for doc in nlp.pipe(textos_limpos, batch_size=256, n_process=1):
            lemmas = [token.lemma_ for token in doc if not token.is_stop]
            lemmatized_results.append(" ".join(lemmas))
            
        chunk['lemmatized_text'] = lemmatized_results
        
        modo = 'w' if i == 0 else 'a'
        cabecalho = True if i == 0 else False
        
        chunk.to_csv(arquivo_saida, mode=modo, header=cabecalho, index=False)
        
        del chunk
        del textos_limpos
        del lemmatized_results
        gc.collect() 
        
        tempo_chunk = time.time() - inicio_chunk
        print(f"Chunk {i+1} processado de forma segura em {tempo_chunk:.2f}s.")

    print(f"\nFinalizado em {(time.time() - inicio_total)/60:.2f} minutos.")

if __name__ == '__main__':
    processar_ambiente_controlado('all_comments_depth_2.csv', 'comentarios_limpos2.csv')