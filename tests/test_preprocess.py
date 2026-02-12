from core_wr_nlp.preprocess import split_reflection_into_sentences

def test_sentence_split_basic():
    text = "This is one. This is two! And three?"
    sents = split_reflection_into_sentences(text)
    assert len(sents) == 3
