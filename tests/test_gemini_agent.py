from types import SimpleNamespace

from nlp_lab.gemini_agent import GeminiToolAgent


class FakeTypes:
    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Tool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AutomaticFunctionCallingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Part:
        @staticmethod
        def from_function_response(**kwargs):
            return {"function_response": kwargs}

    class Content:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


class FakeClient:
    def __init__(self):
        self.calls = []
        function_call = SimpleNamespace(name="search_knowledge", args={"query": "urgência"})
        first = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(
            parts=[SimpleNamespace(function_call=function_call)]))],
            usage_metadata=SimpleNamespace(prompt_token_count=8, candidates_token_count=2))
        second = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))],
            text="A demonstração não faz triagem clínica. Fonte: emergencia.",
            usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=10))
        self.responses = [first, second]
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_tool_call_returns_grounded_source_and_usage():
    client = FakeClient()
    agent = GeminiToolAgent(model="fake", client=client, sdk_types=FakeTypes)
    result = agent.run("O que faço em uma urgência?")
    assert result["route"] == "gemini_tool_agent"
    assert result["tool_calls"] == ["search_knowledge"]
    assert result["sources"] == ["emergencia"]
    assert result["metrics"]["prompt_tokens"] == 20
    assert client.calls[1]["contents"][-1].kwargs["role"] == "tool"


def test_emergency_bypasses_llm_provider():
    client = FakeClient()
    agent = GeminiToolAgent(model="fake", client=client, sdk_types=FakeTypes)
    result = agent.run("Meu cachorro foi atropelado")
    assert result["metrics"]["llm_called"] is False
    assert client.calls == []
