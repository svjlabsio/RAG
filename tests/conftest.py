import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(autouse=False)
def reset_generation_client():
    import rag.generation as gen_module
    gen_module._client = None
    yield
    gen_module._client = None
