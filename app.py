import streamlit as st
import pandas as pd
from textblob import TextBlob
import re
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Texto Simple",
    page_icon="📊",
    layout="wide"
)

# Título
st.title("📝 Analizador de Texto con TextBlob")

st.markdown("""
Esta aplicación utiliza TextBlob para realizar un análisis básico de texto:

- Análisis de sentimiento y subjetividad
- Extracción de palabras clave
- Análisis de frecuencia de palabras
- Identificación de términos importantes con **TF-IDF**
""")

# Barra lateral
st.sidebar.title("Opciones")

modo = st.sidebar.selectbox(
    "Selecciona el modo de entrada:",
    ["Texto directo", "Archivo de texto"]
)

# -----------------------------
# CONTAR PALABRAS
# -----------------------------

def contar_palabras(texto):

    stop_words = set([
        "a","al","algo","algunas","algunos","ante","antes","como","con","contra",
        "cual","cuando","de","del","desde","donde","durante","e","el","ella",
        "ellas","ellos","en","entre","era","eras","es","esa","esas","ese",
        "eso","esos","esta","estas","este","esto","estos","ha","han",
        "hasta","he","la","las","le","les","lo","los","me","mi","mis",
        "mucho","muchos","muy","nada","ni","no","nos","nosotros",
        "nuestra","nuestro","o","otra","otras","otro","otros",
        "para","pero","poco","por","porque","que","quien","se","si",
        "sin","sobre","somos","son","soy","su","sus",
        "también","te","tiene","todo","todos",
        "un","una","unos","y","yo",

        # Inglés
        "a","about","above","after","again","against","all","am","an","and",
        "any","are","as","at","be","because","been","before","being","below",
        "between","both","but","by","could","did","do","does","doing",
        "down","during","each","few","for","from","further","had","has",
        "have","having","he","her","here","hers","him","his","how",
        "i","if","in","into","is","it","its","itself","me","more","most",
        "my","myself","no","nor","not","of","off","on","once","only",
        "or","other","our","ours","ourselves","out","over","own","same",
        "she","should","so","some","such","than","that","the","their",
        "them","themselves","then","there","these","they","this","those",
        "through","to","too","under","until","up","very","was","we",
        "were","what","when","where","which","while","who","why",
        "with","would","you","your","yours","yourself"
    ])

palabras = re.findall(r'\b\w+\b', texto.lower())

    palabras_filtradas = [
        p for p in palabras
        if p not in stop_words and len(p) > 2
    ]

    contador = {}

    for palabra in palabras_filtradas:
        contador[palabra] = contador.get(palabra, 0) + 1

    contador_ordenado = dict(
        sorted(contador.items(), key=lambda x: x[1], reverse=True)
    )

    return contador_ordenado, palabras_filtradas


# -----------------------------
# TRADUCCIÓN
# -----------------------------

def traducir_texto(texto):
    try:
        traduccion = GoogleTranslator(source='auto', target='en').translate(texto)
        return traduccion
    except:
        return texto


# -----------------------------
# TF-IDF
# -----------------------------

def calcular_tfidf(texto):

    vectorizer = TfidfVectorizer(stop_words='english')

    tfidf_matrix = vectorizer.fit_transform([texto])

    palabras = vectorizer.get_feature_names_out()
    valores = tfidf_matrix.toarray()[0]

    tfidf_dict = dict(zip(palabras, valores))

    tfidf_dict = dict(
        sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)
    )

    return tfidf_dict


# -----------------------------
# PROCESAR TEXTO
# -----------------------------

def procesar_texto(texto):

    texto_original = texto
    texto_ingles = traducir_texto(texto)

    blob = TextBlob(texto_ingles)

    sentimiento = blob.sentiment.polarity
    subjetividad = blob.sentiment.subjectivity

    frases = [
        f.strip()
        for f in re.split(r'[.!?]+', texto_original)
        if f.strip()
    ]

    contador_palabras, palabras = contar_palabras(texto_ingles)

    return {
        "sentimiento": sentimiento,
        "subjetividad": subjetividad,
        "frases": frases,
        "contador_palabras": contador_palabras,
        "texto_original": texto_original,
        "texto_traducido": texto_ingles
    }


# -----------------------------
# VISUALIZACIONES
# -----------------------------

def crear_visualizaciones(resultados):

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Sentimiento")

        sentimiento_norm = (resultados["sentimiento"] + 1) / 2

        st.progress(sentimiento_norm)

        st.write(f"Score: {resultados['sentimiento']:.2f}")

        st.subheader("Subjetividad")

        st.progress(resultados["subjetividad"])

        st.write(f"Score: {resultados['subjetividad']:.2f}")

    with col2:

        st.subheader("Palabras más frecuentes")

        palabras_top = dict(
            list(resultados["contador_palabras"].items())[:10]
        )

        st.bar_chart(palabras_top)

    # TF-IDF
    st.subheader("Palabras más importantes (TF-IDF)")

    tfidf = calcular_tfidf(resultados["texto_traducido"])

    tfidf_top = dict(list(tfidf.items())[:10])

    st.bar_chart(tfidf_top)

    # Texto traducido
    st.subheader("Texto traducido")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Original")
        st.text(resultados["texto_original"])

    with col2:
        st.write("Inglés")
        st.text(resultados["texto_traducido"])


# -----------------------------
# INTERFAZ
# -----------------------------

if modo == "Texto directo":

    texto = st.text_area(
        "Ingresa tu texto",
        height=200
    )

    if st.button("Analizar"):

        if texto.strip():

            with st.spinner("Analizando..."):

                resultados = procesar_texto(texto)

                crear_visualizaciones(resultados)

        else:

            st.warning("Ingresa texto para analizar")


elif modo == "Archivo de texto":

    archivo = st.file_uploader(
        "Sube archivo",
        type=["txt","csv","md"]
    )

    if archivo:

        contenido = archivo.getvalue().decode("utf-8")

        if st.button("Analizar archivo"):

            with st.spinner("Analizando..."):

                resultados = procesar_texto(contenido)

                crear_visualizaciones(resultados)


# -----------------------------
# INFORMACIÓN
# -----------------------------

with st.expander("📚 Información sobre el análisis"):

    st.markdown("""
### Sobre el análisis de texto

**Sentimiento**
- -1 → Muy negativo
- 0 → Neutral
- 1 → Muy positivo

**Subjetividad**
- 0 → Objetivo
- 1 → Subjetivo

### TF-IDF

TF-IDF identifica las palabras más importantes en un documento considerando:

- frecuencia dentro del documento
- rareza dentro del corpus

TF-IDF = TF × IDF
""")

st.markdown("---")

st.markdown("Desarrollado con ❤️ usando Streamlit")
