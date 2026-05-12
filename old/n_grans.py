import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 1. Carregar o arquivo já filtrado e com as tags que criamos no passo anterior
df = pd.read_csv('reddit_dados_focados_tags.csv')
minhas_stop_words = list(ENGLISH_STOP_WORDS) + ['nt', 'wo', 'like', 'feel', 'feels', 'sounds', 'people', 'just', 'check_mark_button', '2026']

# 2. Garantir que a coluna lematizada seja tratada como texto puro e remover valores nulos
# Usamos a lematizada porque 'scrolling tiktok' e 'scrolled tiktok' viram a mesma coisa
df['lemmatized_text'] = df['lemmatized_text'].fillna('').astype(str)

# 3. Lista dos temas que criamos anteriormente
temas = [
    'Tema_Cognitivo', 
    'Tema_Mental', 
    'Tema_Comportamento', 
    'Tema_Fisico_Tempo'
]

# 4. Configurar o extrator de N-gramas
# ngram_range=(2, 3) significa que queremos capturar pares e trios de palavras
# stop_words='english' remove palavras de ligação comuns do inglês (and, the, is, at...)
vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words=minhas_stop_words)

# 5. Loop para analisar cada tema separadamente
for tema in temas:
    print(f"\n{'='*40}")
    print(f"🔍 ANALISANDO TEMA: {tema.upper()}")
    print(f"{'='*40}")
    
    # Pega apenas os textos (linhas) que receberam "True" para este tema específico
    textos_do_tema = df[df[tema] == True]['lemmatized_text']
    
    # Se por acaso algum tema não tiver nenhum comentário, o código pula para o próximo
    if len(textos_do_tema) == 0:
        print("Nenhum comentário encontrado para este tema.")
        continue

    # Aplica a extração e contagem
    X = vectorizer.fit_transform(textos_do_tema)
    
    # Soma as frequências de cada N-grama e pega os nomes deles
    frequencias = X.sum(axis=0).A1
    ngrams = vectorizer.get_feature_names_out()
    
    # Cria um DataFrame apenas para organizar os resultados deste tema
    df_ngrams = pd.DataFrame({
        'N-grama': ngrams, 
        'Frequencia': frequencias
    })
    
    # Ordena do mais frequente para o menos frequente e pega os Top 15
    top_15_ngrams = df_ngrams.sort_values(by='Frequencia', ascending=False).head(15)
    
    # Imprime o resultado na tela de forma bonitinha
    print(top_15_ngrams.to_string(index=False))