from spacy.tokens import Token
from wn import Wordnet
from SPARQLWrapper import SPARQLWrapper, JSON
import requests_cache
from os import path
import numpy as np
import torch
import spacy

def load_spacy_model():

    # Check if the spacy model is installed
    model_name = "en_core_web_trf"
    if not spacy.util.is_package(model_name):
        spacy.cli.download(model_name)
    
    # Load the spacy model
    spacy.require_cpu() # .prefer_gpu()
    nlp = spacy.load(model_name)
    
    print('\n' + '-' * 25, 'SPACY MODEL:', torch.cuda.get_device_name(torch.cuda.device_count() - 1) if torch.cuda.is_available() else 'CPU', '-' * 25 + '\n')

    # CUSTOM ATTRIBUTE 1: Get the lexical semantics using Wordnet 
    # ERROR: sqlite3.OperationalError --> download the resource--> python -m wn download oewn:2023
    en = Wordnet('oewn:2023')
    Token.set_extension("lexical_semantics", getter = lambda token: extract_semantics(token, lexicon = en))

    return nlp

def extract_semantics(token: Token, lexicon: Wordnet, max_hypernym_depth = 3) -> str:

    # Skip the tokens that are not (proper) nouns, auxiliaries, or verbs
    pos = {'NOUN': 'n', 'PROPN': 'n', 'VERB': 'v', 'ADJ': 'a'}
    if token.pos_ not in pos.keys():
        return None
    
    # Generate the query using the lemma and remove the possessive form
    query = token.lemma_.replace("'s", '').strip()
    
    # STRAT 1: WordNet
    senses = lexicon.senses(query, pos = pos.get(token.pos_))
    if len(senses) > 0:
        lexical_semantics = senses[0].word().id.split('-')[1] #.replace('oewn-', '').

        # Get the hypernym paths
        hypernym_paths = senses[0].synset().hypernym_paths()
        if len(hypernym_paths) > 0:
            path = [hypernym.words()[0].id for hypernym in hypernym_paths[0]]
            depth = max_hypernym_depth if len(path) >= max_hypernym_depth else len(path)
            return [lexical_semantics] + path[:depth]
        else:
            return [lexical_semantics]
    else:
        
        # STRAT 2: DBpedia
        matches = dbpedia_query(query)
        
        if len(matches) > 0:
            return matches[:max_hypernym_depth + 1]
        else:
            if token.text.strip() != query:
                matches = dbpedia_query(token.text.strip())
                if len(matches) > 0:
                    return matches[:max_hypernym_depth + 1]
            
    return None

def dbpedia_query(query: str) -> list:
    
    # Clean the query
    doc = query.replace("'", '').lower().strip().replace(' ', '_').replace('-', '_')
    
    # Use a cache to avoid multiple requests
    requests_cache.install_cache("_dbcache", expire_after = 60 * 60 * 24 * 5)  # Cache expires after 1 day
    
    # Set the SPARQL endpoint
    sparql = SPARQLWrapper("https://dbpedia.org/sparql", returnFormat= JSON, agent = "Lexical Semantics/0.2")
    
    # Construct a single query to find the resource and its types
    sparql.setQuery(f"""
        SELECT ?item ?type
        WHERE {{
            ?item rdfs:label ?label .
            ?label bif:contains "'{doc}'" .
            ?item rdf:type ?type .
            
            FILTER(lang(?label) = 'en')
            FILTER(
                STRSTARTS(STR(?type), "http://schema.org/") || STRSTARTS(STR(?type), "http://www.w3.org/") 
            )

        }}
        ORDER BY DESC(?score)
        LIMIT 10
    """)

    # Execute the query
    try:
        search_results = sparql.query().convert()
    except Exception as e:
        print(f"Error in querying the DBpedia with '{doc}': {e}")
        return []
    
    # Function to clean the text
    cleanText = lambda t: t.replace(' ', '_').replace('owl#', '').replace('core#', '').lower() 

    # Process the results
    outputs = []
    if search_results["results"]["bindings"]:
        for result in search_results["results"]["bindings"]:
            label = path.basename(result['type']['value'])
            outputs.append(cleanText(label))

    # Return the outputs
    if len(outputs) > 0:
        return [cleanText(doc)] + np.unique(outputs).tolist()
    else:
        return []
    
def merge_entity_tokens(doc, noun_chunks: bool = False):
    """
    Merge the entities in the doc.
    """
    
    # Merge the entities in the doc
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            retokenizer.merge(ent)
    
    # Merge the compound tokens in the doc
    if noun_chunks:
        with doc.retokenize() as retokenizer:
            for chunk in doc.noun_chunks:
                if len([t for t in chunk if t.dep_ in {"amod", "compound"}]) > 0:
                    retokenizer.merge(chunk)

    return doc
    
def linguistic_features(doc, lexical_semantics_level:int = 0, verbose: bool = False):
    
    # Merge the entity tokens in the doc
    doc = merge_entity_tokens(doc, noun_chunks=False)
            
    if verbose:
        print('\n' + '-' * 25, 'DOC:', doc.text, '-' * 25)
        
    # Extract the features from the doc
    features = []
    for token in doc:

        # Get the semantics of the token
        semantics = token._.lexical_semantics
        
        # Save the first semantic feature
        if semantics:
            if token.pos_ in ['AUX', 'VERB']:
                sem = semantics[0]
            else:
                sem = semantics[lexical_semantics_level] if len(semantics) > lexical_semantics_level else semantics[-1]
            features.append(sem.lower())
            
        if verbose:
            print('Token:', token.text, f'({token.pos_})' ,'--> :', semantics)
    features, idx = np.unique(features, return_index=True)
    features = features[np.argsort(idx)].tolist()
    return features

