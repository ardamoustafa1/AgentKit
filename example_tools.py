import asyncio
from agentkit.tools import ToolRegistry, tool, execute_tool
from agentkit.tools.builtins import web_search, python_repl

# Kendi özel tool'umuzu yaratıyoruz
@tool
def hesapla(islem: str, a: float, b: float) -> float:
    """
    İki sayı arasında temel matematiksel işlemleri ('topla', 'cikar', 'carp', 'bol') yapar.
    """
    if islem == "topla":
        return a + b
    elif islem == "cikar":
        return a - b
    elif islem == "carp":
        return a * b
    elif islem == "bol":
        return a / b if b != 0 else 0.0
    return 0.0

async def main() -> None:
    registry = ToolRegistry()
    registry.register(hesapla)
    registry.register(web_search)
    registry.register(python_repl)

    schemas = registry.get_all_schemas()
    import json
    print("--- LLM'e Gidecek Tool Şemaları ---")
    print(json.dumps(schemas[0], indent=2, ensure_ascii=False))

    tool_name_from_llm = "hesapla"
    tool_args_from_llm = {"islem": "carp", "a": 5.5, "b": 2.0}

    target_tool = registry.get_tool(tool_name_from_llm)
    
    if target_tool:
        print("\n--- Tool Çalıştırılıyor ---")
        result = await execute_tool(target_tool, **tool_args_from_llm)
        print(f"Tool Sonucu: {result}")

if __name__ == "__main__":
    asyncio.run(main())
