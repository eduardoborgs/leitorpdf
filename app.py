import streamlit as st
import fitz  # PyMuPDF
from deep_translator import GoogleTranslator
from transformers import pipeline
import torch

st.set_page_config(
    page_title="PDF PowerTool | Tradutor e Resumidor",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def extrair_texto_pdf(file_bytes):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        texto_completo = "".join(pagina.get_text() for pagina in doc)
        doc.close()
        
        if not texto_completo.strip():
            return None, "Erro: O PDF parece conter apenas imagens ou está vazio. Nenhum texto foi extraído."
        
        return texto_completo, None
    except Exception as e:
        if "password" in str(e).lower():
            return None, "Erro: Este PDF está protegido por senha e não pode ser lido."
        else:
            return None, f"Erro inesperado ao processar o PDF: {e}"

def traduzir_texto(texto, idioma_destino='pt'):
    try:
        return GoogleTranslator(source='auto', target=idioma_destino).translate(texto)
    except Exception as e:
        return f"Ocorreu um erro durante a tradução: {e}"

@st.cache_resource
def carregar_modelo_resumo(model_name):
    try:
        summarizer = pipeline("summarization", model=model_name)
        return summarizer
    except Exception as e:
        st.error(f"Erro ao carregar o modelo '{model_name}': {e}")
        return None

def resumir_texto(texto, summarizer):
    try:
        max_chunk_length = 1024
        chunks = [texto[i:i + max_chunk_length] for i in range(0, len(texto), max_chunk_length)]
        
        resumo_final = ""
        for chunk in chunks:
            resumo_parte = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            if resumo_parte:
                resumo_final += resumo_parte[0]['summary_text'] + " "
        
        return resumo_final.strip()
    except Exception as e:
        return f"Ocorreu um erro ao gerar o resumo: {e}"

with st.sidebar:
    st.title("PDF PowerTool")
    st.markdown("---")
    st.header("1. Faça o Upload do PDF")
    arquivo_pdf = st.file_uploader("Selecione o arquivo", type=["pdf"], label_visibility="collapsed")
    st.markdown("---")
    st.subheader("Sobre")
    st.info(
        "Este aplicativo foi criado para facilitar a leitura e o entendimento de documentos em PDF."
        "\n\n**Tecnologias:**\n- Python & Streamlit\n- PyMuPDF\n- Hugging Face Transformers"
    )

st.title(" ferramenta de Tradução e Resumo de PDFs")
st.markdown("Faça o upload de um arquivo PDF na barra lateral para começar.")

if arquivo_pdf is None:
    st.warning("Por favor, carregue um arquivo PDF para habilitar as funcionalidades.")
    st.stop()

pdf_bytes = arquivo_pdf.getvalue()
texto_extraido, erro = extrair_texto_pdf(pdf_bytes)

if erro:
    st.error(erro)
    st.stop()

st.success(f"Texto extraído com sucesso! O documento contém **{len(texto_extraido)}** caracteres.")
with st.expander("Clique para ver o texto completo extraído do PDF"):
    st.text_area("Texto Extraído", texto_extraido, height=250, label_visibility="collapsed")

tab_traducao, tab_resumo = st.tabs(["Tradução do Documento", "Resumo Inteligente"])

with tab_traducao:
    st.header("Traduzir Texto")
    idioma = st.selectbox(
        "Selecione o idioma de destino:",
        ('português', 'inglês', 'espanhol', 'francês', 'alemão'),
        key='lang_select'
    )
    mapa_idiomas = {'português': 'pt', 'inglês': 'en', 'espanhol': 'es', 'francês': 'fr', 'alemão': 'de'}
    
    if st.button("Traduzir Agora", key='translate_btn'):
        with st.spinner(f"Traduzindo para {idioma}... Por favor, aguarde."):
            texto_traduzido = traduzir_texto(texto_extraido, mapa_idiomas[idioma])
        st.text_area("Resultado da Tradução", texto_traduzido, height=300)

with tab_resumo:
    st.header("Resumir Texto")
    modelo_selecionado = st.selectbox(
        "Escolha o modelo de resumo:",
        (
            "Falconsai/text_summarization",  
            "facebook/bart-large-cnn",       
            "google/mt5-small"              
        ),
        index=0, 
        help="Modelos maiores oferecem mais qualidade, mas demoram mais para processar."
    )
    
    if st.button("Gerar Resumo Agora", key='summarize_btn'):
        with st.status("Gerando resumo...", expanded=True) as status:
            status.update(label="Carregando modelo de IA... (pode demorar na primeira vez)", state="running")
            summarizer = carregar_modelo_resumo(modelo_selecionado)
            
            if summarizer:
                status.update(label="Processando o texto e criando o resumo...", state="running")
                resumo = resumir_texto(texto_extraido, summarizer)
                status.update(label="Resumo concluído!", state="complete")
                st.text_area("Resultado do Resumo", resumo, height=300)
            else:
                status.update(label="Falha ao carregar o modelo.", state="error")
