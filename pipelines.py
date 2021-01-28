import spacy
from components.chunker import chunk_component 
from components.mwe import mwe_component

# spacy.require_gpu(0)
_nlp = None

def get_nlp(mwe=True, chunk=True):
    global _nlp
    if _nlp is None:
        _nlp = spacy.load('en_core_web_trf')
        if mwe:
            _nlp.add_pipe('mwe_component', name="mwe_comp", last=True)
        if chunk:
            _nlp.add_pipe("chunk_component", name="chunk_comp", last=True)
    
    return _nlp