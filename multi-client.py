# This code is a Python script that demonstrates how to create a multi-client
# application using the Model Control Protocol (MCP). It allows for
# interaction with multiple MCP servers, each potentially using different
# connection types (SSE or stdio). The script includes functionality to
# initialize clients, retrieve available tools, and handle tool calls.
#
# Code borrowed and hacked from https://modelcontextprotocol.info/docs/tutorials/building-a-client/

import json
from huggingface_hub import get_token
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from typing import Any, List, Dict
import asyncio
import os

# Model ID for the OpenAI model to be used.
MODEL_ID = "gpt-4o-mini"

# Define configurations for multiple clients.
# Here we find a local sse server, and local stdio server as examples
client_configs = [
    {"server_params": "http://localhost:8000/sse", "connection_type": "sse"},
    {"server_params": StdioServerParameters(command="./tools/code-sandbox-mcp/bin/code-sandbox-mcp-darwin-arm64",args=[],env={}), "connection_type": "stdio"},
]

# Prompt for the assistant, including tool descriptions.
SYSTEM_PROMPT = """You are a helpful assistant capable of accessing external functions and engaging in casual chat. Use the responses from these function calls to provide accurate and informative answers. The answers should be natural and hide the fact that you are using tools to access real-time information. Guide the user about available tools and their capabilities. Always utilize tools to access real-time information when required. Engage in a friendly manner to enhance the chat experience.

# Tools

{tools}

# Notes 

- Ensure responses are based on the latest information available from function calls.
- Maintain an engaging, supportive, and friendly tone throughout the dialogue.
- Always highlight the potential of available tools to assist users comprehensively."""
 
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

class MCPClient:
    """
    A client class for interacting with the MCP (Model Control Protocol) server.
    Supports different connection types (SSE or stdio) based on configuration.
    """
    def __init__(self, server_params: str, connection_type: str = "sse"):
        """Initialize the MCP client with server parameters and connection type."""
        self.server_params = server_params
        self.connection_type = connection_type.lower()
        self.session = None
        self._client = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def connect(self):
        """Establish a connection to the MCP server using the specified type."""
        if self.connection_type == "sse":
            self._client = sse_client(self.server_params)
        elif self.connection_type == "stdio":
            self._client = stdio_client(self.server_params)
        else:
            raise ValueError(f"Unsupported connection type: {self.connection_type}")

        self.read, self.write = await self._client.__aenter__()
        session = ClientSession(self.read, self.write)
        self.session = await session.__aenter__()
        await self.session.initialize()

    async def get_available_tools(self) -> List[Any]:
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        response = await self.session.list_tools()
        return response.tools

    def call_tool(self, tool_name: str) -> Any:
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
 
        async def callable(*args, **kwargs):
            response = await self.session.call_tool(tool_name, arguments=kwargs)
            return " ".join([content.text for content in response.content])
 
        return callable

async def agent_loop(query: str, tools: Dict[str, Any], messages: List[dict] = None):
    messages = (
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    tools="\n- ".join(
                        [
                            f"{t['name']}: {t['schema']['function']['description']}"
                            for t in tools.values()
                        ]
                    )
                ),
            },
        ]
        if messages is None
        else messages
    )
    messages.append({"role": "user", "content": query})
 
    first_response = await client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=([t["schema"] for t in tools.values()] if len(tools) > 0 else None),
        max_tokens=4096,
        temperature=0,
    )
    print(f"\n{first_response}\n")
    messages.append(first_response.choices[0].message)
 
    stop_reason = (
        "tool_calls"
        if first_response.choices[0].message.tool_calls is not None
        else first_response.choices[0].finish_reason
    )
 
    if stop_reason == "tool_calls":
        for tool_call in first_response.choices[0].message.tool_calls:
            arguments = (
                json.loads(tool_call.function.arguments)
                if isinstance(tool_call.function.arguments, str)
                else tool_call.function.arguments
            )
            tool_result = await tools[tool_call.function.name]["callable"](**arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(tool_result),
                }
            )
 
        new_response = await client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
        )
 
    elif stop_reason == "stop":
        new_response = first_response
 
    else:
        raise ValueError(f"Unknown stop reason: {stop_reason}")
 
    messages.extend(choice.message for choice in new_response.choices)
    messages.append(
        {"role": "assistant", "content": new_response.choices[0].message.content}
    )
 
    return new_response.choices[0].message.content, messages

def fix_tool_schema(schema: dict) -> dict:
    """
    Recursively fix the tool schema by ensuring that any parameter of type 'array'
    has an 'items' field.
    
    This function specifically looks into the nested "function" key and then into its
    "parameters" properties.
    """
    # Check if the schema is in the expected function format.
    if schema.get("type") == "function" and "function" in schema:
        func = schema["function"]
        if "parameters" in func and isinstance(func["parameters"], dict):
            params = func["parameters"]
            if "properties" in params:
                for prop_name, prop_schema in params["properties"].items():
                    # If the property is an array and no "items" field is defined, add one.
                    if prop_schema.get("type") == "array" and "items" not in prop_schema:
                        prop_schema["items"] = {"type": "string"}
    return schema

async def main():
    """
    In this main function, we demonstrate creating multiple MCP clients,
    each possibly using a different connection type. For instance, one
    client might use SSE while another uses stdio.
    """
 
    # Create MCP client instances concurrently
    clients = {}
    for config in client_configs:
        key = f"{config['connection_type']}_{config['server_params']}"
        clients[key] = MCPClient(**config)
 
    # Initialize all clients concurrently
    await asyncio.gather(*(client.connect() for client in clients.values()))
 
    all_tools = {}
    for key, mcp_client in clients.items():
        tools_list = await mcp_client.get_available_tools()
        all_tools[key] = {
            tool.name: {
                "name": tool.name,
                "callable": mcp_client.call_tool(tool.name),
                "schema": tool.inputSchema and {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                } or None,
            }
            for tool in tools_list if tool.name != "list_tables"
        }

    combined_tools = {}
    for client_key, client_tools in all_tools.items():
        for tool_name, tool_info in client_tools.items():
            # Use the tool's base name as the key
            base_name = tool_info["name"]
            fixed_schema = fix_tool_schema(tool_info["schema"])
            if base_name in combined_tools:
                # Optionally handle duplicate tool names here (e.g. log a warning or merge)
                pass
            combined_tools[base_name] = {
                "name": tool_info["name"],
                "callable": tool_info["callable"],
                "schema": fixed_schema,
            }


    print(combined_tools)

    # Now use the combined_tools dictionary in your agent loop.
    messages = None
    while True:
        try:
            user_input = input("\nEnter your prompt (or 'quit' to exit): ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break

            response, messages = await agent_loop(user_input, combined_tools, messages)
            print("\nResponse:", response)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
