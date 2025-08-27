import streamlit as st
import fitz  # PyMuPDF
from deep_translator import GoogleTranslator
from transformers import pipeline
import torch

st.set_page_config(
    page_title="PDF PowerTool",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'texto_extraido' not in st.session_state:
    st.session_state['texto_extraido'] = None
if 'resultado_processamento' not in st.session_state:
    st.session_state['resultado_processamento'] = None
if 'nome_arquivo' not in st.session_state:
    st.session_state['nome_arquivo'] = None

@st.cache_data
def extrair_texto_pdf(file_bytes):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        texto_completo = "".join(pagina.get_text() for pagina in doc)
        doc.close()
        if not texto_completo.strip():
            return None, "Erro: O PDF parece conter apenas imagens ou está vazio."
        return texto_completo, None
    except Exception as e:
        if "password" in str(e).lower():
            return None, "Erro: Este PDF está protegido por senha."
        return None, f"Erro inesperado ao processar o PDF: {e}"

def traduzir_texto(texto, idioma_destino='pt', container=st):
    try:
        max_chars = 4500
        chunks = [texto[i:i + max_chars] for i in range(0, len(texto), max_chars)]
        texto_traduzido_completo = []
        
        progress_bar = container.progress(0, text="Traduzindo partes do documento...")
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            traducao_chunk = GoogleTranslator(source='auto', target=idioma_destino).translate(chunk)
            if traducao_chunk:
                texto_traduzido_completo.append(traducao_chunk)
            progress_bar.progress((i + 1) / total_chunks, text=f"Traduzindo parte {i+1}/{total_chunks}")
        
        progress_bar.empty()
        return "".join(texto_traduzido_completo)
    except Exception as e:
        return f"Ocorreu um erro durante o processo de tradução: {e}"

@st.cache_resource
def carregar_modelo_resumo(model_name):
    return pipeline("summarization", model=model_name)

def resumir_texto(texto, _summarizer):
    try:
        max_chunk_length = 1024
        chunks = [texto[i:i + max_chunk_length] for i in range(0, len(texto), max_chunk_length)]
        resumo_final = ""
        for chunk in chunks:
            resumo_parte = _summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            if resumo_parte:
                resumo_final += resumo_parte[0]['summary_text'] + " "
        return resumo_final.strip()
    except Exception as e:
        return f"Ocorreu um erro ao gerar o resumo: {e}"

with st.sidebar:
    st.header("1. Carregue seu PDF")
    arquivo_pdf = st.file_uploader(
        "Arraste e solte ou clique para carregar", 
        type=["pdf"], 
        label_visibility="collapsed"
    )

    if arquivo_pdf:
        if st.session_state.get('nome_arquivo') != arquivo_pdf.name:
            st.session_state['nome_arquivo'] = arquivo_pdf.name
            st.session_state['resultado_processamento'] = None 
            with st.spinner("Processando PDF..."):
                pdf_bytes = arquivo_pdf.getvalue()
                texto, erro = extrair_texto_pdf(pdf_bytes)
                if erro:
                    st.error(erro)
                    st.session_state['texto_extraido'] = None
                else:
                    st.success(f"Arquivo '{arquivo_pdf.name}' processado!")
                    st.session_state['texto_extraido'] = texto

st.header("Ferramenta de Tradução e Resumo de PDFs")

if not st.session_state['texto_extraido']:
    st.info("⬅️ Comece carregando um arquivo PDF na barra lateral.")
    st.stop()

col_controles, col_resultados = st.columns((1, 1.2)) 

with col_controles:
    st.subheader("Suas Ações")
    
    with st.expander("Ver texto original do PDF"):
        st.text_area(
            "Texto Extraído", 
            st.session_state['texto_extraido'], 
            height=200, 
            label_visibility="collapsed"
        )

    tab_traducao, tab_resumo = st.tabs(["Tradução", "Resumo"])

    with tab_traducao:
        st.markdown("Traduzir Documento")
        idioma = st.selectbox(
            "Selecione o idioma de destino:",
            ('português', 'inglês', 'espanhol', 'francês', 'alemão'),
            key='lang_select'
        )
        if st.button("Executar Tradução", type="primary"):
            mapa_idiomas = {'português': 'pt', 'inglês': 'en', 'espanhol': 'es', 'francês': 'fr', 'alemão': 'de'}
            resultado = traduzir_texto(st.session_state['texto_extraido'], mapa_idiomas[idioma], col_resultados)
            st.session_state['resultado_processamento'] = resultado
    
    with tab_resumo:
        st.markdown("Gerar Resumo Inteligente")
        modelo = st.selectbox(
            "Escolha o modelo de IA:",
            ("Falconsai/text_summarization", "facebook/bart-large-cnn", "google/mt5-small"),
            help="Modelos maiores podem ter mais qualidade, mas demoram mais."
        )
        if st.button("Executar Resumo", type="primary"):
            with st.spinner("Carregando modelo e gerando resumo... Aguarde."):
                summarizer = carregar_modelo_resumo(modelo)
                if summarizer:
                    resultado = resumir_texto(st.session_state['texto_extraido'], summarizer)
                    st.session_state['resultado_processamento'] = resultado
                else:
                    st.session_state['resultado_processamento'] = "Erro: Não foi possível carregar o modelo de resumo."

with col_resultados:
    st.subheader("Área de Resultados")
    
    if st.session_state['resultado_processamento']:
        st.text_area(
            "Resultado", 
            st.session_state['resultado_processamento'], 
            height=400, 
            label_visibility="collapsed"
        )
        if st.button("🗑️ Limpar Resultado"):
            st.session_state['resultado_processamento'] = None
            st.experimental_rerun() 
    else:
        st.info("Selecione uma ação no painel à esquerda e clique em 'Executar' para ver o resultado aqui.")
