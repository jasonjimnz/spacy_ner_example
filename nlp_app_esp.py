import spacy
import random
from spacy.training import Example
from typing import Union, Dict, List

LANGUAGE = 'es_core_news_sm'
TEXT = "Este verano, quiero ir a visitar la Alhambra"
TRAINING_TEXTS = [
    (
        # Texto que añadiremos como entrenamiento
        "En Granada se encuentra situada la Alhambra",
        # Lista de tuplas que contiene: Posición de inicio de la etiqueta, posición final, y nombre de la etiqueta
        [(35, 43, "lugar_de_interes")]
    ),
    (
        "El último fin de semana quisimos visitar el Alcázar de Segovia, pero el mal tiempo nos hizo posponerlo",
        [(44, 51, "lugar_de_interes")]
    ),
    (
        "Queríamos ir a Cuenca, pero se nos estropeó el coche y no pudimos visitar las Casas Colgadas, iremos el mes que viene",
        [(78, 92, "lugar_de_interes")]
    ),
    (
        "Las construcción de la Sagrada Familia está alargándose demasiado, a ver cuando la terminan para ir a Barcelona",
        [(23, 38, "lugar_de_interes")]
    )
]
nlp = spacy.load(LANGUAGE)


def parse_tokens(processed_text: spacy.tokens.Doc) -> List[Dict[str, Union[str, bool]]]:
    return [{
        'text': token.text,
        'lemma': token.lemma_,
        'pos': token.pos_,
        'tag': token.tag_,
        'shape': token.shape_,
        'stopword': token.is_stop,
        'punctuation': token.is_punct,
        'whitespace': token.is_space,
        'ent_type': token.ent_type_
    } for token in processed_text]


def training_model(spacy_model: spacy.Language) -> None:
    n_iterations = 10
    for s_iter in range(n_iterations):
        random.shuffle(TRAINING_TEXTS)
        for raw_text, entity_positions in TRAINING_TEXTS:
            doc = spacy_model.make_doc(raw_text)
            example = Example.from_dict(doc, {"entities": entity_positions})
            spacy_model.update([example])
            # Para guardar el modelo en disco
            # spacy_model.to_disk('RUTA_DEL_MODELO')


tokens = nlp(TEXT)
print("Modelo antes de ser entrenado\n")
for t in parse_tokens(tokens):
    print(t)

training_model(spacy_model=nlp)
tokens2 = nlp(TEXT)
print("\nModelo después de ser entrenado\n")
for t in parse_tokens(tokens2):
    print(t)
