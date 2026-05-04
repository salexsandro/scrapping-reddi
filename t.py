import pandas as pd

df = pd.read_csv('comentarios_limpos2.csv')

# ADICIONE ESTA LINHA: Ela vai imprimir no terminal exatamente 
# quais colunas o Pandas conseguiu enxergar no seu arquivo.
print("Colunas encontradas:", df.columns.tolist())

# O código para e mostra o erro aqui
df['cleaned_text'] = df['cleaned_text'].astype(str)