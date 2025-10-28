import stanza

stanza.download("en")  # run once

nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,ner,sentiment")

def annotate_stanza(text: str):
    doc = nlp(text)
    results = []

    for i, sent in enumerate(doc.sentences, start=1):
        sent_info = {
            "sentence_index": i,
            "text": sent.text,
            "sentiment": sent.sentiment,  # typically 0 = negative, 1 = neutral, 2 = positive
            "tokens": []
        }

        for word in sent.words:
            sent_info["tokens"].append({
                "text": word.text,
                "lemma": word.lemma,
                "pos": word.upos,
                "xpos": word.xpos,
                "ner": None  # placeholder, filled below if entity exists
            })

        # Named entities are separate spans, so we attach them afterward
        for ent in sent.ents:
            for token in sent_info["tokens"]:
                if token["text"] in ent.text.split():
                    token["ner"] = ent.type

        results.append(sent_info)

    return results
