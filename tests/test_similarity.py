import numpy as np
from core_wr_nlp.similarity import cosine_similarity_matrix

def test_cosine_similarity_matrix_shape():
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0]], dtype=np.float32)
    sim = cosine_similarity_matrix(a, b)
    assert sim.shape == (2, 1)
    assert np.isclose(sim[0, 0], 1.0)
    assert np.isclose(sim[1, 0], 0.0)
