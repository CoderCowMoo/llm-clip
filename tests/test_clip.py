import llm


def clip_test_run_embedding():
    model = llm.get_embedding_model("clip")
    result = model.embed("bunny")
    assert len(result) == 512
    assert isinstance(result[0], float)

def siglip_test_run_embedding():
    model = llm.get_embedding_model("siglip")
    result = model.embed("bunny")
    assert len(result) == 1152
    assert isinstance(result[0], float)