import httpx
from typing import List, Dict, Any
from agentkit.tools.base import ToolDefinition

def _build_json_schema(params: List[Dict[str, Any]]) -> Dict[str, Any]:
    """OpenAPI parameters listesini JSON Schema formatına çevirir."""
    schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    for p in params:
        name = p.get("name")
        schema["properties"][name] = {
            "type": p.get("schema", {}).get("type", "string"),
            "description": p.get("description", "")
        }
        if p.get("required", False):
            schema["required"].append(name)
    return schema

def import_openapi(spec_url: str) -> List[ToolDefinition]:
    """OpenAPI (Swagger) şemasını indirip AgentKit araçlarına dönüştürür."""
    # Sadece senkron olarak spec'i indirip araçları üreteceğiz,
    # Araçların KENDİLERİ asenkron olarak HTTP isteği atacak.
    response = httpx.get(spec_url)
    response.raise_for_status()
    spec = response.json()
    
    base_url = spec.get("servers", [{"url": ""}])[0].get("url", "")
    if not base_url.startswith("http"):
        base_url = "/".join(spec_url.split("/")[:3]) + base_url
        
    tools = []
    
    paths = spec.get("paths", {})
    for path, methods in paths.items():
        for method, details in methods.items():
            if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                continue
                
            operation_id = details.get("operationId", f"{method}_{path.replace('/', '_')}".strip("_"))
            description = details.get("summary", details.get("description", f"Executes {method.upper()} on {path}"))
            
            parameters = details.get("parameters", [])
            schema = _build_json_schema(parameters)
            
            # Dinamik fonksiyon oluşturucu (Closure ile verileri yakala)
            def create_dynamic_func(req_method: str, req_path: str, req_params: List[Dict[str, Any]]):
                async def dynamic_request(**kwargs):
                    async with httpx.AsyncClient() as client:
                        # Path variables (e.g. /users/{id})
                        url = req_path
                        query_params = {}
                        for p in req_params:
                            name = p.get("name")
                            if name in kwargs:
                                if p.get("in") == "path":
                                    url = url.replace(f"{{{name}}}", str(kwargs[name]))
                                elif p.get("in") == "query":
                                    query_params[name] = kwargs[name]
                        
                        full_url = f"{base_url.rstrip('/')}/{url.lstrip('/')}"
                        
                        try:
                            # Basitlik adına GET ağırlıklı API'ler için tasarlandı. 
                            # Body gerekirse burada kwargs işlenebilir.
                            resp = await client.request(req_method.upper(), full_url, params=query_params)
                            resp.raise_for_status()
                            return resp.text
                        except Exception as e:
                            return f"API isteği başarısız oldu: {str(e)}"
                return dynamic_request

            # Tool ismini optimize et (Alfanumerik ve alt çizgi)
            import re
            tool_name = re.sub(r'[^a-zA-Z0-9_]', '_', operation_id)
            
            tool = ToolDefinition(
                name=tool_name,
                description=description,
                parameters=schema,
                func=create_dynamic_func(method, path, parameters)
            )
            tools.append(tool)
            
    return tools
