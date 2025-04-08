# agent-mcp

These two examples demonstrates how to create a multi-client application using the Model Control Protocol (MCP). 

```
uv run -m multi-client

Enter your prompt (or 'quit' to exit): what tools?

Response: I have access to a variety of tools that can help with different tasks:

1. **Database Operations**: 
   - Connect to a PostgreSQL database.
   - Execute read-only SQL queries.
   - Analyze SQL queries with execution plans.

2. **File Management**:
   - Copy files to and from a sandboxed filesystem.
   - Write files directly to the sandbox.

3. **Sandboxed Environment**:
   - Initialize a compute environment for code execution.
   - Execute shell commands in a sandboxed environment.
   - Stop and remove the sandboxed environment.

If you have a specific task in mind, feel free to ask, and I can guide you on how to use these tools!

Enter your prompt (or 'quit' to exit): 
```

They allow for interaction with multiple MCP servers, each potentially using different connection types (either SSE or stdio). 

The script is a REPL that includes functionality to initialize clients, retrieve available tools, and handle tool calls.

This code is borrowed and hacked from https://modelcontextprotocol.info/docs/tutorials/building-a-client/