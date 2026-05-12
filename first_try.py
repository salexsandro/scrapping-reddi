import pandas as pd
import re
from transformers import pipeline
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────
INPUT_CSV = "./all_comments_depth_2.csv"
OUTPUT_CSV = "./dataset_analisado.csv"

# Palavras que não agregam ao sentido do sofrimento/sintomas
# Elas são removidas apenas na contagem final das palavras-chave do tópico
DOMAIN_STOPWORDS = [
    "tiktok", "reels", "shorts", "youtube", "video", "videos", 
    "app", "phone", "scroll", "scrolling", "doomscrolling", 
    "reddit", "instagram", "social", "media", "just", "like", "im"
]

# ─────────────────────────────────────────────
# 1. CARREGAMENTO E LIMPEZA LEVE (RegEx)
# ─────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove URLs (http...)
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove marcações do Reddit
    text = re.sub(r'\[deleted\]|\[removed\]|\[excluído\]', '', text)
    # Remove quebras de linha excessivas
    text = re.sub(r'\n+', ' ', text)
    # Remove espaços duplos
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("Carregando e limpando os dados...")
df = pd.read_csv(INPUT_CSV)
df = df.head(100)

# ---------------- AJUSTE AQUI ----------------
# Se o CSV não tiver a coluna 'title' (ex: arquivo só de comentários), 
# o script cria uma coluna vazia temporária para não quebrar.
if 'title' not in df.columns:
    df['title'] = ""
if 'text' not in df.columns:
    df['text'] = ""
# ---------------------------------------------

# Combina Título e Texto 
df['full_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
df['full_text'] = df['full_text'].apply(clean_text)

# Remove linhas onde o texto ficou vazio após a limpeza
df = df[df['full_text'].str.len() > 10].reset_index(drop=True)

# ─────────────────────────────────────────────
# 2. ANÁLISE DE SENTIMENTO (RoBERTa)
# ─────────────────────────────────────────────
print("\nIniciando Análise de Sentimento (isso pode demorar dependendo da sua CPU/GPU)...")

# Usa um modelo treinado especificamente em redes sociais (Twitter/X e Reddit)
# Labels de saída: 'positive', 'neutral', 'negative'
sentiment_model = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
    max_length=512,       # Limita o tamanho do texto para o modelo não estourar a memória
    truncation=True,       # Corta textos muito longos
    device=0
)

def get_sentiment(text):
    try:
        result = sentiment_model(text)[0]
        return result['label'], result['score']
    except Exception as e:
        return "error", 0.0

# Aplica a função de sentimento (pode levar tempo)
sentiments = df['full_text'].apply(get_sentiment)
df['sentiment_label'] = [s[0] for s in sentiments]
df['sentiment_score'] = [s[1] for s in sentiments]

print("Análise de Sentimento concluída!")

# ─────────────────────────────────────────────
# 3. MODELAGEM DE TÓPICOS (BERTopic)
# ─────────────────────────────────────────────
print("\nIniciando Modelagem de Tópicos (BERTopic)...")

docs = df['full_text'].tolist()

# a) Modelo de Embedding (Transforma texto em números preservando contexto)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# b) Vetorizador (Aqui aplicamos nossa lista de Stopwords customizada)
# O texto contido em "docs" não perde a palavra "tiktok", 
# mas o algoritmo ignora "tiktok" na hora de nomear o tópico.
vectorizer_model = CountVectorizer(stop_words='english')
# Adicionando nossas stopwords customizadas no vocabulário de ignorados
vectorizer_model.stop_words_ = vectorizer_model.get_stop_words().union(DOMAIN_STOPWORDS)

# c) Treinamento do BERTopic
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    language="english",
    calculate_probabilities=False,
    min_topic_size=15 # Agrupa no mínimo 15 relatos para formar um assunto. Ajuste conforme o tamanho do dataset.
)

# Ajusta o modelo e descobre os tópicos
topics, probs = topic_model.fit_transform(docs)

# Salva os resultados no Dataframe
df['topic_id'] = topics

print("\nTópicos encontrados!")
print(topic_model.get_topic_info()[['Topic', 'Count', 'Name']].head(10))

# ─────────────────────────────────────────────
# 4. SALVAR RESULTADOS
# ─────────────────────────────────────────────
# Junta as informações do tópico de volta no Dataframe para facilitar a leitura
topic_info = topic_model.get_topic_info()
df = df.merge(topic_info[['Topic', 'Name']], left_on='topic_id', right_on='Topic', how='left')
df = df.drop(columns=['Topic'])
df.rename(columns={'Name': 'topic_name'}, inplace=True)

df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"\nTodos os dados salvos em: {OUTPUT_CSV}")

# Se quiser visualizar um gráfico interativo no Jupyter/Colab:
# fig = topic_model.visualize_topics()
# fig.show()