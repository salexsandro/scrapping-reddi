
import pandas as pd
df = pd.read_csv('comentarios_limpos2.csv')

df['cleaned_text'] = df['cleaned_text'].astype(str)

dicionarios = {
    'Tema_Cognitivo': 'attention span|focus|brain fog|goldfish|memory|brain rot|fried|rewired|cant read|brain dead|zombie|numb',
    'Tema_Mental': 'anxiety|anxious|depressed|depression|fomo|compare|jealous|inadequate|overwhelmed|lonely|isolation|stress|guilt|shame',
    'Tema_Comportamento': 'doomscrolling|doom scroll|rabbit hole|binge|loop|trap|hooked|addicted|addiction|cant stop|mindless|autopilot|muscle memory|deleted|uninstalled',
    'Tema_Fisico_Tempo': 'sleep|insomnia|bedtime|stayed up|hours wasted|procrastinate|eyes hurt|tired|exhausted|sedentary|time blind|lost track'
}

for tema, palavras_chave in dicionarios.items():
    df[tema] = df['cleaned_text'].str.contains(palavras_chave, case=False, na=False, regex=True)

colunas_temas = list(dicionarios.keys())

df_focado = df[df[colunas_temas].any(axis=1)].copy()

linhas_originais = len(df)
linhas_filtradas = len(df_focado)
linhas_removidas = linhas_originais - linhas_filtradas

print(f"Total de comentários originais: {linhas_originais}")
print(f"Comentários removidos (sem tags): {linhas_removidas}")
print(f"Comentários mantidos (relevantes): {linhas_filtradas}")

df_focado.to_csv('reddit_dados_focados_tags.csv', index=False)

print("\nArquivo 'reddit_dados_focados_tags.csv' salvo com sucesso!")