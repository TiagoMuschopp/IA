import pandas as pd
import mysql.connector
from sklearn.cluster import KMeans
import pickle
import os

# 1. Conectar ao banco de dados MySQL
def conectar_mysql():
    return mysql.connector.connect(
        host='localhost',      # ou seu host
        user='root',           # ou seu usuário
        password='Senha@2024', # ou sua senha
        database='CYBERPIRATAS'
    )

# 2. Carregar os dados
def carregar_dados():
    conexao = conectar_mysql()
    query = "SELECT * FROM VENDAS"
    df = pd.read_sql(query, conexao)
    conexao.close()
    return df

# 3. Processar os dados
def processar_dados(df):
    if 'DT_REGISTRO' not in df.columns or 'RENDA' not in df.columns or 'VALOR' not in df.columns:
        raise ValueError("As colunas necessárias ('DT_REGISTRO', 'RENDA', 'VALOR') não estão presentes no banco de dados.")
    df['DT_REGISTRO'] = pd.to_datetime(df['DT_REGISTRO'])
    df['Ano'] = df['DT_REGISTRO'].dt.year
    df['Mes'] = df['DT_REGISTRO'].dt.month
    return df[['CD_PESSOA_FISICA', 'RENDA', 'VALOR', 'PRODUTO', 'SEXO', 'IDADE', 'Ano', 'Mes']].copy()

# 4. Realizar segmentação
def segmentar_clientes(df):
    modelo = KMeans(n_clusters=3, random_state=42)
    df['Segmento'] = modelo.fit_predict(df[['RENDA', 'VALOR', 'IDADE']])
    salvar_modelo(modelo, 'models/segmentacao.pkl')
    return df

# 5. Salvar o modelo treinado
def salvar_modelo(modelo, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, 'wb') as f:
        pickle.dump(modelo, f)

# 6. Criar recomendações baseadas no sexo, idade e faixa de preço
def recomendar_produtos(df, idade_cliente, sexo_cliente, preco_min, preco_max):
    # Filtra os dados para encontrar clientes com a mesma idade e sexo
    clientes_similares = df[(df['IDADE'] == idade_cliente) & (df['SEXO'] == sexo_cliente)]

    if clientes_similares.empty:
        print(f"Não foram encontrados clientes com a idade {idade_cliente} e sexo {sexo_cliente}.")
        return []

    # Filtra os produtos pela faixa de preço
    produtos_filtrados = clientes_similares[(clientes_similares['VALOR'] >= preco_min) & (clientes_similares['VALOR'] <= preco_max)]

    if produtos_filtrados.empty:
        print(f"Não foram encontrados produtos dentro da faixa de preço {preco_min} - {preco_max}.")
        return []

    # Conta a frequência dos produtos comprados dentro da faixa de preço
    produtos_comprados = produtos_filtrados['PRODUTO'].value_counts()

    # Recomendação: retorna os produtos mais comprados dentro da faixa de preço
    recomendacoes = produtos_comprados.head(5).index.tolist()  # Top 5 produtos mais comprados

    return recomendacoes

# 7. Função principal
def main():
    try:
        # Solicitar ao usuário idade, sexo e faixa de preço antes de carregar e processar os dados
        idade_cliente = int(input("Digite a idade do cliente: "))
        sexo_cliente = input("Digite o sexo do cliente (M/F): ").upper()

        # Validar sexo
        if sexo_cliente not in ['M', 'F']:
            print("Sexo inválido! Use 'M' para Masculino ou 'F' para Feminino.")
            return

        # Solicitar a faixa de preço
        preco_min = float(input("Digite o preço mínimo: "))
        preco_max = float(input("Digite o preço máximo: "))

        # Carregar dados diretamente do MySQL
        dados = carregar_dados()
        
        # Processar dados
        dados_processados = processar_dados(dados)
        
        # Realizar segmentação
        resultados = segmentar_clientes(dados_processados)
        
        # Exibir resultados
        print(resultados)
        
        # Salvar resultados em CSV
        caminho_resultados = 'data/vendas_segmentadas.csv'
        resultados.to_csv(caminho_resultados, index=False)
        print(f"Resultados salvos em: {caminho_resultados}")
        
        # Gerar recomendações baseadas na idade, sexo e faixa de preço fornecidos
        recomendacoes = recomendar_produtos(resultados, idade_cliente, sexo_cliente, preco_min, preco_max)
        if recomendacoes:
            print(f"Recomendações para um cliente de {sexo_cliente}, {idade_cliente} anos, na faixa de preço de {preco_min} a {preco_max}: {recomendacoes}")
        else:
            print("Nenhuma recomendação disponível para os parâmetros fornecidos.")
    
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == '__main__':
    main()
