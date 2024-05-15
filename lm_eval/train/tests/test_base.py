import pytest

@pytest.fixture(scope="session")
def model(request):
    return request.param

@pytest.mark.parametrize("model", ["model1", "model2", "model3"], indirect=True)
def test1(model):
    pass

@pytest.mark.parametrize("model", ["model1", "model2", "model3"], indirect=True)
def test2(model):
    pass