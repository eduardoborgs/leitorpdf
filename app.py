import streamlit as st
import fitz  # PyMuPDF
from deep_translator import GoogleTranslator
from transformers import pipeline
import torch # Necessário para o pipeline do transformers

# --- FUNÇÕES CORE ---

# Função para extrair texto de um PDF
# @st.cache_data # Opcional: Usar cache para não reprocessar o mesmo arquivo
def extrair_texto_pdf(arquivo_pdf):
    """
    Extrai o texto de todas as páginas de um arquivo PDF.
    """
    try:
        doc = fitz.open(stream=arquivo_pdf.read(), filetype="pdf")
        texto_completo = ""
        for pagina in doc:
            texto_completo += pagina.get_text()
        doc.close()
        return texto_completo
    except Exception as e:
        st.error(f"Erro ao ler o PDF: {e}")
        return None

# Função para traduzir o texto
def traduzir_texto(texto, idioma_destino='pt'):
    """
    Traduz o texto para o idioma de destino usando o Google Translate.
    """
    try:
        return GoogleTranslator(source='auto', target=idioma_destino).translate(texto)
    except Exception as e:
        st.error(f"Erro na tradução: {e}")
        return "Não foi possível traduzir o texto."

# Função para resumir o texto
# Usaremos um modelo multilingual que funciona bem com português.
# @st.cache_resource # Cache para o modelo de IA não ser carregado toda hora
def resumir_texto(texto):
    """
    Resume o texto usando um modelo da Hugging Face.
    """
    try:
        # Carrega o modelo de sumarização (pode demorar na primeira vez)
        summarizer = pipeline("summarization", model="google/mt5-small", tokenizer="google/mt5-small")
        # Dividir o texto em pedaços menores se for muito grande
        max_chunk_length = 1024
        chunks = [texto[i:i+max_chunk_length] for i in range(0, len(texto), max_chunk_length)]
        
        resumo_final = ""
        for chunk in chunks:
            resumo_parte = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            if resumo_parte:
                resumo_final += resumo_parte[0]['summary_text'] + " "
        
        return resumo_final.strip()
    except Exception as e:
        st.error(f"Erro ao resumir: {e}")
        return "Não foi possível gerar o resumo."

# O resto do código da interface Streamlit virá aqui...
# --- INTERFACE STREAMLIT ---

st.set_page_config(page_title="PDF PowerTool", layout="wide")

st.title(" ferramenta de Tradução e Resumo de PDFs")
st.markdown("Faça o upload de um arquivo PDF e escolha a ação desejada.")

# Colunas para organizar a interface
col1, col2 = st.columns(2)

with col1:
    st.header("1. Faça o Upload do seu PDF")
    arquivo_pdf = st.file_uploader("Selecione o arquivo PDF", type=["pdf"])
    
    texto_extraido = ""
    if arquivo_pdf is not None:
        with st.spinner("Extraindo texto do PDF..."):
            texto_extraido = extrair_texto_pdf(arquivo_pdf)
        if texto_extraido:
            st.success("Texto extraído com sucesso!")
            with st.expander("Ver texto extraído"):
                st.text_area("", texto_extraido, height=250)

with col2:
    st.header("2. Escolha a Ação")
    if texto_extraido:
        acao = st.selectbox("Selecione uma ação:", ["", "Traduzir", "Resumir"])

        # --- Lógica de Tradução ---
        if acao == "Traduzir":
            idioma = st.selectbox(
                "Selecione o idioma de destino:",
                ('português', 'inglês', 'espanhol', 'francês')
            )
            mapa_idiomas = {'português': 'pt', 'inglês': 'en', 'espanhol': 'es', 'francês': 'fr'}
            
            if st.button("Traduzir Texto"):
                with st.spinner(f"Traduzindo para {idioma}..."):
                    texto_traduzido = traduzir_texto(texto_extraido, mapa_idiomas[idioma])
                st.subheader("Resultado da Tradução")
                st.text_area("", texto_traduzido, height=300)

        # --- Lógica de Resumo ---
        elif acao == "Resumir":
            if st.button("Gerar Resumo"):
                with st.spinner("Criando resumo... Isso pode levar alguns minutos."):
                    resumo = resumir_texto(texto_extraido)
                st.subheader("Resumo do Texto")
                st.text_area("", resumo, height=300)

st.sidebar.title("Sobre")
st.sidebar.info(
    "Este aplicativo foi desenvolvido para facilitar a leitura e compreensão de documentos PDF."
    "\n\n**Tecnologias utilizadas:**"
    "\n- Python"
    "\n- Streamlit"
    "\n- PyMuPDF"
    "\n- Hugging Face Transformers"
)