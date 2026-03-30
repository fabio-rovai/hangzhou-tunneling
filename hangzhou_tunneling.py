#!/usr/bin/env python3
"""
Hangzhou Tunneling — Qwen-Agent powered document evaluation with BITF verification.

Three agents (Decomposer, Scorer, Verifier) orchestrate through the
brain-in-the-fish Rust MCP server. No Claude dependency.

Usage:
    python hangzhou_tunneling.py evaluate README.md --intent "assess documentation quality"
    python hangzhou_tunneling.py evaluate tender.pdf --intent "assess methodology"
"""

import json
import subprocess
import time
import select
import sys
import os
from typing import Optional

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

# Path to the BITF MCP server binary
BITF_SERVER = os.environ.get(
    "BITF_SERVER",
    os.path.expanduser("~/.cargo/shared-target/release/brain-in-the-fish-mcp")
)


# =============================================================================
# MCP Client — manages persistent connection to the Rust server
# =============================================================================

class MCPClient:
    """Persistent JSON-RPC connection to the brain-in-the-fish MCP server."""

    def __init__(self, server_path: str = BITF_SERVER):
        self.server_path = server_path
        self.proc = None
        self._id = 0

    def start(self):
        """Start the MCP server and complete the handshake."""
        self.proc = subprocess.Popen(
            [self.server_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # Initialize
        resp = self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hangzhou-tunneling", "version": "1.0"}
        })
        # Send initialized notification
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        time.sleep(0.3)
        return resp

    def stop(self):
        """Stop the MCP server."""
        if self.proc:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=5)
            self.proc = None

    def tool(self, name: str, arguments: dict) -> dict:
        """Call an MCP tool and return the parsed result."""
        resp = self._call("tools/call", {"name": name, "arguments": arguments})
        if resp and "result" in resp:
            text = resp["result"].get("content", [{}])[0].get("text", "{}")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
        elif resp and "error" in resp:
            return {"error": resp["error"].get("message", str(resp["error"]))}
        return {"error": "no response"}

    def _call(self, method: str, params: dict) -> Optional[dict]:
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        self._send(msg)
        return self._recv()

    def _send(self, msg: dict):
        line = json.dumps(msg) + "\n"
        self.proc.stdin.write(line.encode())
        self.proc.stdin.flush()

    def _recv(self, timeout: float = 15.0) -> Optional[dict]:
        time.sleep(1.0)
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if ready:
            line = self.proc.stdout.readline().decode().strip()
            if line:
                return json.loads(line)
        return None


# Global MCP client instance
_mcp: Optional[MCPClient] = None


def get_mcp() -> MCPClient:
    global _mcp
    if _mcp is None:
        _mcp = MCPClient()
        _mcp.start()
    return _mcp


# =============================================================================
# BITF Tools — registered with qwen-agent
# =============================================================================

@register_tool('bitf_ingest')
class BITFIngest(BaseTool):
    description = 'Ingest a document into the BITF evaluation server. Call this first.'
    parameters = {
        'type': 'object',
        'properties': {
            'path': {'type': 'string', 'description': 'Path to the document file'},
            'intent': {'type': 'string', 'description': 'What to evaluate (e.g., "assess documentation quality")'},
        },
        'required': ['path', 'intent']
    }

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        result = get_mcp().tool("eval_ingest", args)
        return json.dumps(result, ensure_ascii=False)


@register_tool('bitf_criteria')
class BITFCriteria(BaseTool):
    description = 'Load evaluation criteria framework. Call after ingest.'
    parameters = {
        'type': 'object',
        'properties': {
            'framework': {'type': 'string', 'description': 'Framework name: generic, academic, tender, clinical'},
        },
        'required': ['framework']
    }

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        result = get_mcp().tool("eval_criteria", args)
        return json.dumps(result, ensure_ascii=False)


@register_tool('bitf_spawn')
class BITFSpawn(BaseTool):
    description = 'Spawn evaluator agent panel. Call after criteria.'
    parameters = {
        'type': 'object',
        'properties': {
            'intent': {'type': 'string', 'description': 'Evaluation intent'},
        },
        'required': ['intent']
    }

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        result = get_mcp().tool("eval_spawn", args)
        return json.dumps(result, ensure_ascii=False)


@register_tool('bitf_load_turtle')
class BITFLoadTurtle(BaseTool):
    description = 'Load OWL Turtle (document decomposition) into the GraphStore. The Turtle must contain typed argument nodes with exact source quotes.'
    parameters = {
        'type': 'object',
        'properties': {
            'turtle': {'type': 'string', 'description': 'OWL Turtle string'},
        },
        'required': ['turtle']
    }

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        result = get_mcp().tool("eds_load_turtle", args)
        return json.dumps(result, ensure_ascii=False)


@register_tool('bitf_record_score')
class BITFRecordScore(BaseTool):
    description = 'Record an LLM score for a criterion.'
    parameters = {
        'type': 'object',
        'properties': {
            'agent_id': {'type': 'string', 'description': 'Agent ID'},
            'criterion_id': {'type': 'string', 'description': 'Criterion ID'},
            'score': {'type': 'number', 'description': 'Score (0-10)'},
            'max_score': {'type': 'number', 'description': 'Max possible score'},
            'justification': {'type': 'string', 'description': 'Why this score'},
        },
        'required': ['agent_id', 'criterion_id', 'score', 'max_score', 'justification']
    }

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        args.setdefault("round", 1)
        args.setdefault("evidence_used", [])
        args.setdefault("gaps_identified", [])
        result = get_mcp().tool("eval_record_score", args)
        return json.dumps(result, ensure_ascii=False)


@register_tool('bitf_score')
class BITFScore(BaseTool):
    description = 'Get the full BITF verdict: SPARQL rules, structural score, gate comparison. Call this last.'
    parameters = {
        'type': 'object',
        'properties': {
            'agent_id': {'type': 'string', 'description': 'Agent ID'},
        },
        'required': ['agent_id']
    }

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        result = get_mcp().tool("eds_score", args)
        return json.dumps(result, ensure_ascii=False)


# =============================================================================
# Agents
# =============================================================================

DECOMPOSER_PROMPT = """You are the Document Decomposer for the Hangzhou Tunneling pipeline.

Your job: read a document and decompose it into an OWL Turtle argument ontology.

IMPORTANT: Do NOT score nodes. Only identify, type, connect, and quote them.

For every factual claim, evidence, citation, and structural element, create a typed node.

Node types: arg:Thesis, arg:SubClaim, arg:Evidence, arg:QuantifiedEvidence, arg:Citation, arg:Counter, arg:Rebuttal, arg:Structural

Requirements:
1. Every node MUST have arg:hasText with the EXACT quote from the document
2. Every claim must be connected via arg:supports, arg:counters, or arg:rebuts
3. Do NOT assign scores — the Rust engine handles scoring

After decomposition, call bitf_load_turtle with your Turtle string.

Schema:
@prefix arg: <http://brain-in-the-fish.dev/arg/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> ."""

SCORER_PROMPT = """You are the Document Scorer for the Hangzhou Tunneling pipeline.

Your job: score the document against each evaluation criterion on a 0-10 scale.

For each criterion:
1. Read the document carefully
2. Assess quality against the criterion
3. Call bitf_record_score with your score and justification
4. Reference specific content from the document

Be rigorous. Use exact quotes as evidence."""

VERIFIER_PROMPT = """You are the Document Verifier for the Hangzhou Tunneling pipeline.

Your job: get the final BITF verdict by calling bitf_score.

The Rust engine will:
1. Run SPARQL rules on the ontology (mining strong/weak/unsupported claims)
2. Compute a structural score from topology (no LLM input)
3. Compare the structural score against the LLM scores via the gate
4. Return CONFIRMED, FLAGGED, or REJECTED

Report the verdict with the full audit trail."""


def create_agents(llm_cfg: dict):
    decomposer = Assistant(
        llm=llm_cfg,
        name='Decomposer',
        system_message=DECOMPOSER_PROMPT,
        function_list=['bitf_ingest', 'bitf_criteria', 'bitf_spawn', 'bitf_load_turtle']
    )
    scorer = Assistant(
        llm=llm_cfg,
        name='Scorer',
        system_message=SCORER_PROMPT,
        function_list=['bitf_record_score']
    )
    verifier = Assistant(
        llm=llm_cfg,
        name='Verifier',
        system_message=VERIFIER_PROMPT,
        function_list=['bitf_score']
    )
    return decomposer, scorer, verifier


# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline(document_path: str, intent: str, llm_cfg: dict):
    """Run the full Hangzhou Tunneling pipeline."""

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  HANGZHOU TUNNELING — Qwen Agents + BITF Verification      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Start MCP server
    print("\nStarting BITF MCP server...")
    mcp = get_mcp()

    # Phase 1: Ingest + Decompose
    print("\n── Phase 1: Ingest & Decompose ─────────────────────────────")
    r = mcp.tool("eval_ingest", {"path": os.path.abspath(document_path), "intent": intent})
    print(f"   Ingested: {r.get('sections', '?')} sections, {r.get('triples_loaded', '?')} triples")

    r = mcp.tool("eval_criteria", {"framework": "generic"})
    criteria = r.get("criteria", [])
    print(f"   Criteria: {len(criteria)} loaded")

    r = mcp.tool("eval_spawn", {"intent": intent})
    agents = r.get("agents", [])
    agent_id = agents[0]["id"] if agents else None
    print(f"   Agents: {len(agents)} spawned, primary: {agent_id}")

    # Create Qwen agents
    decomposer, scorer, verifier = create_agents(llm_cfg)

    # Run decomposer
    print("\n   Running Qwen Decomposer agent...")
    doc_text = open(document_path).read()
    decompose_msg = [{"role": "user", "content": f"Decompose this document into an OWL Turtle argument ontology. Use EXACT quotes only.\n\nDocument:\n{doc_text[:8000]}"}]

    turtle_result = None
    for response in decomposer.run(messages=decompose_msg):
        if isinstance(response, list):
            for msg in response:
                content = msg.get("content", "")
                if "turtle" in content.lower() and "arg:" in content:
                    turtle_result = content

    if turtle_result:
        r = mcp.tool("eds_load_turtle", {"turtle": turtle_result})
        print(f"   Turtle loaded: {r.get('node_count', '?')} nodes, {r.get('triples_loaded', '?')} triples")
    else:
        print("   ⚠ Decomposer did not produce Turtle — using fallback")

    # Phase 2: Score
    print("\n── Phase 2: Score ──────────────────────────────────────────")
    score_msg = [{"role": "user", "content": f"Score this document against these criteria: {json.dumps([c['title'] for c in criteria])}. The agent_id is {agent_id}. The criterion IDs are: {json.dumps({c['title']: c['id'] for c in criteria})}.\n\nDocument:\n{doc_text[:8000]}"}]

    for response in scorer.run(messages=score_msg):
        pass  # Scorer calls bitf_record_score via tools
    print("   Scores recorded")

    # Phase 3: Verify
    print("\n── Phase 3: Verify ─────────────────────────────────────────")
    verify_msg = [{"role": "user", "content": f"Get the BITF verdict. The agent_id is {agent_id}. Call bitf_score."}]

    for response in verifier.run(messages=verify_msg):
        if isinstance(response, list):
            for msg in response:
                content = msg.get("content", "")
                if "CONFIRMED" in content or "FLAGGED" in content or "REJECTED" in content:
                    print(f"   {content[:200]}")

    # Direct verdict call as fallback
    r = mcp.tool("eds_score", {"agent_id": agent_id})
    print(f"\n   Structural score: {r.get('structural_score', '?')}")
    topo = r.get("topology", {})
    if topo:
        print(f"   Nodes: {topo.get('node_count', '?')} | Evidence: {topo.get('evidence_count', '?')} | Claims: {topo.get('claim_count', '?')}")
    mined = r.get("mined_facts", {})
    if mined:
        print(f"   Mined: strong={mined.get('strong_claims', '?')} unsupported={mined.get('unsupported_claims', '?')}")
    verdict = r.get("verdict", "")
    print(f"\n   ╔═══════════════════════════════════════════╗")
    print(f"   ║  {str(verdict)[:50]}")
    print(f"   ╚═══════════════════════════════════════════╝")

    mcp.stop()

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  PIPELINE COMPLETE                                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] != "evaluate":
        print("Usage: python hangzhou_tunneling.py evaluate <document> --intent <intent>")
        print()
        print("Options:")
        print("  --model <name>     Qwen model (default: Qwen3-8B)")
        print("  --server <url>     Model server URL (default: http://localhost:8000/v1)")
        print("  --dashscope        Use DashScope API instead of local server")
        sys.exit(1)

    document = sys.argv[2]
    intent = "assess document quality"
    model = "Qwen3-8B"
    server_url = "http://localhost:8000/v1"
    use_dashscope = False

    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--intent" and i + 1 < len(sys.argv):
            intent = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--server" and i + 1 < len(sys.argv):
            server_url = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--dashscope":
            use_dashscope = True
            i += 1
        else:
            i += 1

    if use_dashscope:
        llm_cfg = {
            'model': model if model != "Qwen3-8B" else 'qwen-max-latest',
            'model_type': 'qwen_dashscope',
            'generate_cfg': {'top_p': 0.8}
        }
    else:
        llm_cfg = {
            'model': model,
            'model_server': server_url,
            'api_key': 'EMPTY',
            'generate_cfg': {'top_p': 0.8}
        }

    run_pipeline(document, intent, llm_cfg)
