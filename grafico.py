import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Leia o seu arquivo CSV
df = pd.read_csv('dataset_analisado_100.csv')

# 2. FILTRANDO O TÓPICO -1 (OU QUALQUER OUTRO INDESEJADO)
# Isso cria um novo dataframe contendo apenas as linhas onde o tópico NÃO é o especificado
df_filtrado = df[df['topic_name'] != '-1_time_social_like_media']

# Outra opção: Se você quiser remover APENAS os outliers (se o topic_id for sempre -1)
# df_filtrado = df[df['topic_id'] != -1]

# 3. Configurações de estilo
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- GRÁFICO 1: Distribuição de Sentimentos ---
sns.countplot(data=df_filtrado, x='sentiment_label', palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('Distribuição Geral de Sentimentos (Sem Ruído)', fontsize=14)
axes[0, 0].set_xlabel('Sentimento')
axes[0, 0].set_ylabel('Quantidade de Comentários')

# --- GRÁFICO 2: Tópicos Mais Discutidos ---
top_topics = df_filtrado['topic_name'].value_counts().nlargest(10).index
sns.countplot(data=df_filtrado, y='topic_name', order=top_topics, palette='magma', ax=axes[0, 1])
axes[0, 1].set_title('Top 10 Tópicos Reais Mais Discutidos', fontsize=14)
axes[0, 1].set_xlabel('Quantidade')
axes[0, 1].set_ylabel('Tópico')

# --- GRÁFICO 3: Distribuição de Scores do Modelo por Sentimento ---
sns.boxplot(data=df_filtrado, x='sentiment_label', y='sentiment_score', palette='coolwarm', ax=axes[1, 0])
axes[1, 0].set_title('Score de Confiança do Modelo por Sentimento', fontsize=14)
axes[1, 0].set_xlabel('Sentimento')
axes[1, 0].set_ylabel('Score (0.0 a 1.0)')

# --- GRÁFICO 4: Engajamento (Upvotes) Médio por Tópico ---
# O .copy() evita avisos do pandas na hora de criar a nova coluna
df_filtrado = df_filtrado.copy()
df_filtrado['log_score'] = np.log1p(df_filtrado['score'])

sns.barplot(data=df_filtrado, y='topic_name', x='log_score', palette='cubehelix', ax=axes[1, 1], errorbar=None)
axes[1, 1].set_title('Engajamento (Log Score) por Tópico', fontsize=14)
axes[1, 1].set_xlabel('Score Médio de Upvotes (Escala Log)')
axes[1, 1].set_ylabel('')

# Ajusta o layout e salva a imagem
plt.tight_layout()
plt.savefig('dashboard_filtrado.png', dpi=300)
plt.show()