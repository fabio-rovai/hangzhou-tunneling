#!/usr/bin/env python3
"""
Hangzhou Tunneling - Minimal qwen-agent example with custom MCP tool
"""

import json
import subprocess
from typing import Any

from qwen_agent.agents import Assistant, GroupChat
from qwen_agent.tools.base import BaseTool, register_tool


# =============================================================================
# Custom Tool: BITF (Brain In The Fish) MCP Server Caller
# =============================================================================
@register_tool('bitf_mcp')
class BITFMCPTool(BaseTool):
    """
    Custom tool that calls the brain-in-the-fish MCP server via JSON-RPC stdin/stdout.
    
    Fill in the BITF logic in the `call` method below.
    """
    description = 'Verify document evaluation results using the brain-in-the-fish MCP server'
    parameters = {
        'type': 'object',
        'properties': {
            'method': {
                'type': 'string',
                'description': 'MCP method to call (e.g., "verify", "validate")'
            },
            'params': {
                'type': 'object',
                'description': 'Parameters to pass to the MCP method'
            }
        },
        'required': ['method', 'params']
    }

    def call(self, params: str, **kwargs) -> str:
        """
        Call the BITF MCP server via JSON-RPC over stdin/stdout.
        
        Args:
            params: JSON string with 'method' and 'params' keys
            **kwargs: Additional context
        
        Returns:
            JSON string with the MCP server response
        """
        # Parse incoming parameters
        request = json.loads(params)
        method = request['method']
        method_params = request['params']
        
        # Build JSON-RPC 2.0 request
        json_rpc_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": method_params
        }
        
        # TODO: Fill in your BITF MCP server logic here
        # Example pattern using subprocess:
        #
        # process = subprocess.Popen(
        #     ['cargo', 'run', '--manifest-path', '/path/to/brain-in-the-fish/Cargo.toml'],
        #     stdin=subprocess.PIPE,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        #     text=True
        # )
        # stdout, stderr = process.communicate(
        #     input=json.dumps(json_rpc_request)
        # )
        # response = json.loads(stdout)
        # return json.dumps(response.get('result', {}), ensure_ascii=False)
        
        # Placeholder response - replace with actual BITF call
        response = {
            "verified": True,
            "message": f"BITF verification placeholder for method: {method}",
            "data": method_params
        }
        
        return json.dumps(response, ensure_ascii=False)


# =============================================================================
# Multi-Agent Setup: Decomposer, Scorer, Verifier
# =============================================================================

def create_decomposer_agent(llm_cfg: dict) -> Assistant:
    """Agent that decomposes documents into evaluable components."""
    return Assistant(
        llm=llm_cfg,
        name='Decomposer',
        system_message="""You are a Document Decomposer. Your role is to:
1. Analyze incoming documents
2. Break them down into evaluable components/sections
3. Identify key criteria for evaluation
4. Pass the decomposed structure to the Scorer""",
        function_list=['bitf_mcp']
    )


def create_scorer_agent(llm_cfg: dict) -> Assistant:
    """Agent that scores document components."""
    return Assistant(
        llm=llm_cfg,
        name='Scorer',
        system_message="""You are a Document Scorer. Your role is to:
1. Receive decomposed document components
2. Evaluate each component against defined criteria
3. Assign scores and provide justification
4. Pass scores to the Verifier""",
        function_list=['bitf_mcp']
    )


def create_verifier_agent(llm_cfg: dict) -> Assistant:
    """Agent that verifies scores using BITF MCP server."""
    return Assistant(
        llm=llm_cfg,
        name='Verifier',
        system_message="""You are a Document Verifier. Your role is to:
1. Receive scores from the Scorer
2. Use the bitf_mcp tool to verify scores against the brain-in-the-fish MCP server
3. Confirm or flag discrepancies
4. Output final verified evaluation""",
        function_list=['bitf_mcp']  # This agent uses the BITF tool
    )


def create_group_chat(llm_cfg: dict) -> GroupChat:
    """Create a multi-agent workflow with 3 agents."""
    agents = [
        create_decomposer_agent(llm_cfg),
        create_scorer_agent(llm_cfg),
        create_verifier_agent(llm_cfg)
    ]
    
    return GroupChat(
        agents=agents,
        llm=llm_cfg
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    # Configure LLM - replace with your model configuration
    llm_cfg = {
        # Option 1: DashScope (Qwen)
        # 'model': 'qwen-max-latest',
        # 'model_type': 'qwen_dashscope',
        
        # Option 2: OpenAI-compatible (vLLM, Ollama, etc.)
        'model': 'Qwen3-8B',
        'model_server': 'http://localhost:8000/v1',
        'api_key': 'EMPTY',
        
        'generate_cfg': {
            'top_p': 0.8
        }
    }
    
    # Option A: Single agent with BITF tool (simplest)
    print("=== Single Agent Demo ===")
    bot = Assistant(
        llm=llm_cfg,
        system_message="You are a document evaluator. Use the bitf_mcp tool for verification.",
        function_list=['bitf_mcp']
    )
    
    # Test the custom tool directly
    print("\nTesting BITF tool directly:")
    test_params = json.dumps({
        "method": "verify",
        "params": {"document_id": "test-123", "score": 0.85}
    })
    result = BITFMCPTool().call(test_params)
    print(f"BITF Response: {result}")
    
    # Option B: Multi-agent workflow
    print("\n=== Multi-Agent Workflow Demo ===")
    group_chat = create_group_chat(llm_cfg)
    
    # Example usage (uncomment to run with actual LLM):
    # messages = [{'role': 'user', 'content': 'Evaluate this document: ...'}]
    # for response in group_chat.run(messages=messages):
    #     print(response)


if __name__ == '__main__':
    main()
