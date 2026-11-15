# Expose local note search through stdio MCP

Local coding assistants launch a read-only MCP search server over stdio, reusing Hybrid retrieval and returning ranked note chunks with provenance. The assistant generates the answer. This replaces the remote authenticated HTTP interface in ADR-0006: a same-machine demonstration no longer requires an identity provider, public endpoint, or tunnel. Remote access and multi-user authentication are not supported. The process runs with the user's local permissions; returned evidence enters the client context and may be sent to its model provider.
