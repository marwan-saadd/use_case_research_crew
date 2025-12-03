"""Standalone Notion MCP connectivity check (no CrewAI dependency)."""

import argparse
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Optional

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

DOCKER_COMMAND = [
    "docker",
    "run",
    "-i",
    "--rm",
    "-e",
    "INTERNAL_INTEGRATION_TOKEN",
    "mcp/notion",
]


async def run_client(token: str, list_only: bool) -> None:
    server_params = StdioServerParameters(
        command=DOCKER_COMMAND[0],
        args=DOCKER_COMMAND[1:],
        env={"INTERNAL_INTEGRATION_TOKEN": token},
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)
        await session.initialize()

        print("Connected. Listing available tools...", flush=True)
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            desc = tool.description or "No description"
            print(f"- {tool.name}: {desc}")

        if list_only:
            return

        target_tool = "notion.list_workspaces"
        if any(tool.name == target_tool for tool in tools_result.tools):
            print(f"Calling {target_tool} ...", flush=True)
            result = await session.call_tool(target_tool, arguments={})
            print("Tool call result:")
            print(json.dumps(result.model_dump(), indent=2))
        else:
            print(f"{target_tool} not exposed by this server; skipping tool call.")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list available tools without calling any.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    token = os.getenv("INTERNAL_INTEGRATION_TOKEN")
    if not token:
        print("INTERNAL_INTEGRATION_TOKEN env var is required", file=sys.stderr)
        sys.exit(1)

    try:
        anyio.run(run_client, token, args.list_only)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
