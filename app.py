from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
import gradio as gr
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os

# API KEY do GROQ
api_key = os.getenv("secret_key")

# Configuração inicial do QP
llm = Groq(model="llama-3.3-70b-versatile", api_key=api_key)

def descricao_colunas(df):
    descricao = '\n'.join([f"`{col}`:{str(df[col].dtype)}" for col in df.columns])
    return 'Aqui estão os detalhes das colunas do DataFrame:\n' + descricao

# Instruções para orientar o modelo
instruction_str = (
    "1. Converta a consulta para código Python executável usando Pandas.\n"
    "2. A linha final do código deve ser uma expressão Python que possa ser chamada com a função `eval()`.\n"
    "3. O código deve representar uma solução para a consulta.\n"
    "4. IMPRIMA APENAS A EXPRESSÃO.\n"
    "5. Não coloque a expressão entre aspas.\n")

# Prompts
pandas_prompt_str = (
    "Você está trabalhando com um dataframe do pandas em Python chamado `df`.\n"
    "{colunas_detalhes}\n\n"
    "Este é o resultado de `print(df.head())`:\n"
    "{df_str}\n\n"
    "Siga estas instruções:\n"
    "{instruction_str}\n"
    "Consulta: {query_str}\n\n"
    "Expressão:"
)

response_synthesis_prompt_str = (
  "Dada uma pergunta de entrada, atue como analista de dados e elabore uma resposta a partir dos resultados da consulta.\n"
  "Responda de forma natural, sem introduções como 'A resposta é:' or something similar.\n"
  "Consulta: {query_str}\n\n"
  "Instruções do Pandas (opcional):\n{pandas_instructions}\n\n"
  "Saída do Pandas: {pandas_output}\n\n"
  "Resposta:"
  "Ao final, exibir o código usado para gerar a resposta, no formato: O código utilizado foi {pandas_instructions}"
)
# Componentes (These need to be outside the function to be accessible)
pandas_prompt = PromptTemplate(pandas_prompt_str)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
llm = Groq(model='llama-3.1-8b-instant', api_key=api_key)


def run_pipeline(query_str: str, df, verbose: bool = True): # Pass df as an argument
    # Input Component
    # Re-format the pandas_prompt here to include the current df details
    formatted_pandas_prompt = pandas_prompt.partial_format(
        instruction_str=instruction_str,
        colunas_detalhes=descricao_colunas(df),
        df_str=df.head(5)
    ).format(query_str=query_str)

    pandas_code_response = llm.complete(formatted_pandas_prompt).text
    # Re-initialize the parser with the current df
    current_pandas_output_parser = PandasInstructionParser(df)
    pandas_result = current_pandas_output_parser.parse(pandas_code_response)
    pandas_result_str = str(pandas_result)
    formatted_synthesis_prompt = response_synthesis_prompt.format(
        query_str=query_str,
        pandas_instructions=pandas_code_response,
        pandas_output=pandas_result_str
    )
    final_response = llm.complete(formatted_synthesis_prompt)

    return final_response.text

## Funções de processamento de dados da página ##
def carregar_dados(caminho_arquivo, df_estado):
  if caminho_arquivo is None or caminho_arquivo == '':
    return "Por favor, faça o upload de um arquivo CSV para analisar", pd.DataFrame(), df_estado
  try:
    df = pd.read_csv(caminho_arquivo)
    return "Arquivo carregado com sucesso!", df.head(), df
  except Exception as e:
    return f"Erro ao carregar o arquivo: {str(e)}", pd.DataFrame(), df_estado

def processar_pergunta(pergunta, df_estado):
  if df_estado is not None and pergunta:
    # Pass df_estado to run_pipeline
    resposta = run_pipeline(query_str=pergunta, df=df_estado)
    return resposta

  return ""

def add_historico(pergunta, resposta, historico_estado):
  if pergunta and resposta:
    historico_estado.append((pergunta, resposta))
    gr.Info('Adicionado ao PDF!', duration=2)
    return historico_estado

def gerar_pdf(historico_estado):
  if not historico_estado:
    return "Não há dados para gerar o PDF", None

  timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
  caminho_pdf = f'relatorio_perguntas_respostas_ {timestamp}.pdf'
  pdf = FPDF()
  pdf.add_page()
  pdf.set_auto_page_break(auto=True,margin=15)
  pdf.set_font('Arial', '', 12)


  for pergunta, resposta in historico_estado:
    pdf.set_font('Arial', 'B', 14)
    pdf.multi_cell(0, 8, txt=pergunta)
    pdf.ln(2)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 8, txt=resposta)
    pdf.ln(6)

  pdf.output(caminho_pdf)
  return caminho_pdf

def limpar_pergunta_resposta():
  return "", ""

def resetar_aplicacao():
  return None, "A aplicação foi resetada. Por favor, faça o upload de um novo arquivo CSV.", pd.DataFrame(), "", None, [], None

## Criação da página utilizando o tema Soft do gradio ##
with gr.Blocks(theme='Soft') as app:
  # Título do app
  gr.Markdown('# Analisando os dados')

  # Descrição do app
  gr.Markdown('''
    Carregue um arquivo CSV e faça perguntas sobre os dados. A cada pergunta, você poderá
    visualizar a resposta e, se desejar, adicionar essa interação ao PDF final, basta clicar
    em "Adicionar ao histórico do PDF". Para fazer uma nova pergunta, clique em "Limpar
    pergunta e resultado". Após definir as perguntas e respostas no histórico, clique em
    "Gerar PDF". Assim, será possível baixar um PDF com o registro completo das suas interações.
    Se você quiser analisar um novo dataset, basta clicar em "Quero analisar outro dataset" ao final da página.
  ''')



  ## Componentes da página ##
  input_arquivo = gr.File(file_count='single',type='filepath',label='Upload CSV')
  upload_status = gr.Textbox(label='Status do Upload')
  tabela_dados = gr.Dataframe()

 # Exemplos de perguntas
  gr.Markdown("""
    Exemplos de perguntas:
    1. Qual é o número de registros no arquivo?
    2. Quais são os tipos de dados das colunas?
    3. Quais são as estatísticas descritivas das colunas numéricas?
  """)

  input_pergunta = gr.Textbox(label='Digite sua pergunta sobre os dados')
  botao_submeter = gr.Button('Enviar')
  output_resposta = gr.Textbox(label='Reposta')

  with gr.Row():
    botao_limpeza = gr.Button('Limpar pergunta e resposta')
    botao_add_pdf = gr.Button('Adicionar ao Histórico do PDF')
    botao_gerar_pdf = gr.Button('Gerar PDF')
  arquivo_pdf = gr.File(label='Download do PDF')
  botao_resetar = gr.Button('Quero analisar outro dataset')

  # Gerenciamento do estado
  df_estado = gr.State(value=None)
  historico_estado = gr.State(value=[])


  # Ações dos componentes da página
  input_arquivo.change(fn=carregar_dados,
                       inputs=[input_arquivo, df_estado],
                       outputs=[upload_status, tabela_dados, df_estado])
  botao_submeter.click(fn=processar_pergunta,
                       inputs=[input_pergunta, df_estado],
                       outputs=[output_resposta])
  botao_limpeza.click(fn=limpar_pergunta_resposta,
                      inputs=[],
                      outputs=[input_pergunta, output_resposta])
  botao_add_pdf.click(fn=add_historico,
                      inputs=[input_pergunta, output_resposta, historico_estado],
                      outputs=[historico_estado])
  botao_gerar_pdf.click(fn=gerar_pdf,
                        inputs=[historico_estado],
                        outputs=[arquivo_pdf])
  botao_resetar.click(fn=resetar_aplicacao,
                      inputs=[],
                      outputs=[input_arquivo, upload_status, tabela_dados, input_pergunta, output_resposta, historico_estado, arquivo_pdf])


if __name__ == "__main__":
    app.launch()