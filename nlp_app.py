import spacy

EN_LANGUAGE = 'en_core_web_sm'
ES_LANGUAGE = 'es_core_news_sm'

en_nlp = spacy.load(EN_LANGUAGE)
es_nlp = spacy.load(ES_LANGUAGE)

EN_TEXT = ("Hi there, I'm just writing a Medium article "
           "about natural language processing")
ES_TEXT = ("Hola, estoy escribiendo un artículo en Mediuim "
           "sobre el procesamiento del lenguaje natural")

en_tokens = en_nlp(EN_TEXT)
es_tokens = es_nlp(ES_TEXT)


def parse_tokens(processed_text: spacy.tokens.Doc) -> None:
    for token in processed_text:
        print({
            'text': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'tag': token.tag_,
            'shape': token.shape_,
            'stopword': token.is_stop,
            'punctuation': token.is_punct,
            'whitespace': token.is_space
        })


print('English!\n')
parse_tokens(en_tokens)
print('\nEspañol!\n')
parse_tokens(es_tokens)
