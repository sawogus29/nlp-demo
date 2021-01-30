from spacy.tokens import Token, Span, Doc
from spacy.language import Language
import logging
import torch
import numpy as np
from transformers import BertTokenizerFast, BertModel, BertForTokenClassification, BertConfig
from functools import reduce


@Language.component("mwe_component")
def mwe_component(doc):
    def set_mwe(sent):
        if len(sent) <= 1 and sent[0].is_space:
            return
        spacy_tokens = [token.text for token in sent]
        pt_input_ids, pt_attention_masks, alignments = tensor_from_list(spacy_tokens, tknz, device)
        assert len(sent) == len(alignments)
        labels = predict(pt_input_ids, pt_attention_masks)

        raw_iobs = get_raw_iobs(labels, alignments)
        for token, raw_iob in zip(sent, raw_iobs):
            token._.mwe_raw_iob = raw_iob

        valid_labels = validate_iob(labels)
        print(valid_labels)

        iobs = align_labels(valid_labels, alignments)
        for token, iob in zip(sent, iobs):
            token._.mwe_iob = iob
        
        gather_MWEs(sent)
        
    for sent in doc.sents:
        set_mwe(sent)

    return doc

def get_inner_len(nested_list):
    for inner_list in nested_list:
        yield len(inner_list)

def slice_by_size(li, sizes):
    i = 0
    for l in sizes:
        yield li[i:i+l]
        i += l

def get_raw_iobs(labels, alignments):
    assert len(labels) == reduce(lambda acc, y: acc+len(y), alignments, 0)
    aligned = []
    for lbs, model_tokens in zip(slice_by_size(labels, get_inner_len(alignments)), alignments):
        aligned.append(list(zip(model_tokens, lbs)))
    return aligned

def align_labels(labels, alignments):
    def current_next(li, next_last=None):
        n = next(li, None)
        while True:
            c = n
            n = next(li, None)
            if n is None:
                break
            yield c, n
        yield c, next_last
    
    def get_most_important_label(lbs):
        priority = ['B', 'I', 'b', 'i', 'o', 'O', '0']
        m_i = priority.index('O') 
        for l in lbs:
            if priority.index(l) < m_i:
                m_i = priority.index(l)
        return priority[m_i]

    assert len(labels) == reduce(lambda acc, y: acc+len(y), alignments, 0)
    aligned = []
    
    for c, n in current_next(slice_by_size(labels, get_inner_len(alignments)), 'O'):
        mil = get_most_important_label(c)
        if mil == 'B':
            if not n or n[0] == 'O':
                aligned.append('O')
            else:
                aligned.append('B')
        else:
            aligned.append(mil)

    return aligned

def validate_iob(iobs):
    valid_iobs = []
    buffer = []
    prev = 'O'
    buffer_mode = False

    valid_after = {
        'O': ['B', 'O', '0'],
        'B': ['I', 'o', 'b'],
        'I': ['I', 'o', 'b', 'O'],
        'o': ['I', 'b', 'i'],
        'b': ['I', 'i', 'o'],
        'i': ['i', 'o', 'I'],
        '0': ['O']
    }
    for tag in iobs + ['O']:
        ############Pre#################
        is_valid = tag in valid_after[prev]
        #0
        if not buffer_mode and is_valid:
            pass
        
        if not buffer_mode and not is_valid:
            valid_iobs.append('O')
            prev = 'O'
            continue
        
        #1
        if buffer_mode and not is_valid:
            # buffer overwrite
            valid_iobs += ['O'] * len(buffer)
            buffer = []
        
        #2
        is_end = tag in ('B', 'O')
        if buffer_mode and is_valid and is_end:
            # buffer flush
            valid_iobs += buffer
            buffer = []

        if buffer_mode and is_valid and not is_end:
            buffer.append(tag)
            prev = tag
            continue

        ############Post#################
        #0
        if not buffer_mode and is_valid:
            if tag == 'B':
                buffer_mode = True
                buffer.append(tag)
            else:
                valid_iobs.append(tag)
            prev = tag
            continue

        #1
        if buffer_mode and not is_valid:
            if tag == 'B':
                buffer_mode = True
                buffer.append(tag)
                prev = tag
                continue
            else:
                buffer_mode = False
                valid_iobs.append(tag)
                prev = 'O'
                continue

        #2
        if buffer_mode and is_valid and is_end:
            if tag == 'B':
                buffer_mode = True
                buffer.append(tag)
                prev = tag
                continue
            else:
                buffer_mode = False
                valid_iobs.append(tag)
                prev = tag
                continue

    if len(valid_iobs) > len(iobs):
        assert len(valid_iobs) == len(iobs)+1
        valid_iobs = valid_iobs[:-1]

    return valid_iobs

def tensor_from_list(token_list, tokenizer, device):
    token_ids, alignments = tokenize_from_list(token_list, tokenizer)
    pt_input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
    pt_attention_masks = torch.ones_like(pt_input_ids).to(device)
    return pt_input_ids, pt_attention_masks, alignments 

def tokenize_from_list(token_list, tokenizer):
    aligns = []
    token_ids = [tknz.cls_token_id]
    for token in token_list:
        tokens = tokenizer.tokenize(token)
        aligns.append(tokens)
        token_ids += tokenizer.convert_tokens_to_ids(tokens)
    token_ids.append(tknz.sep_token_id)
    return token_ids, aligns

def predict(pt_input_ids, pt_attention_masks):
    pt_input_ids.to(device)
    pt_attention_masks.to(device)
    
    model.eval()
    
    with torch.no_grad():
        outputs = model(pt_input_ids, token_type_ids=None,
                        attention_mask=pt_attention_masks)
    
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    token_labels = np.argmax(logits, axis=2).squeeze().tolist()
    token_labels = [id_to_label[id] for id in token_labels]
    
    return token_labels[1:-1]

def get_mwes(span):
    mwes = set() 
    for token in span:
        if token._.mwe is not None:
            mwes.add(tuple(token._.mwe))
    return mwes
        
def get_model_and_tokenizer():
    output_dir = './model_save/'
    global _tknz, _model
    if _tknz is None:
        _tknz = BertTokenizerFast.from_pretrained("bert-base-uncased")
    if _model is None:
        _model = BertForTokenClassification.from_pretrained(output_dir, num_labels=len(label_map)+1)
        _model.to(device)
    
    return _model, _tknz

def gather_MWEs(sent):
    #TODO: Although validate_iob() is not perpect,
    # gather_MWEs() suppose it is perfect for convinience.
    BI = []
    bi = []
    for token in sent:
        if token._.mwe_iob == 'B':
            if BI:
                BI = []
            BI.append(token)
            token._.mwe = BI

        elif token._.mwe_iob == 'I':
            assert BI
            BI.append(token)
            token._.mwe = BI

        elif token._.mwe_iob == 'b':
            if bi:
                bi = []
            bi.append(token)
            token._.mwe = bi
        
        elif token._.mwe_iob == 'i':
            assert bi
            bi.append(token)
            token._.mwe = bi

label_map = {'O':0, 'B':1, 'I':2, '0':3, 'o':4,'b':5, 'i':6}
id_to_label = {0:'O', 1:'B', 2:'I', 3:'0', 4:'o', 5:'b', 6:'i'}

Token.set_extension('mwe_iob', default='O', force=True)
Token.set_extension('mwe_raw_iob', default=None, force=True)
Token.set_extension('mwe', default=None, force=True)
Span.set_extension('mwes', getter=get_mwes, force=True)
Doc.set_extension('mwes', getter=get_mwes, force=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model, _tknz = None, None
model, tknz = get_model_and_tokenizer()
