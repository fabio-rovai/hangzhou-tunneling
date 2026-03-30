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
import re
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
        if not os.path.exists(self.server_path):
            raise FileNotFoundError(
                f"BITF server not found at {self.server_path}\n"
                "Build it: cd brain-in-the-fish && cargo build --release\n"
                "Or set BITF_SERVER=/path/to/brain-in-the-fish-mcp"
            )
        self.proc = subprocess.Popen(
            [self.server_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        resp = self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hangzhou-tunneling", "version": "1.0"}
        })
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        time.sleep(0.3)
        return resp

    def stop(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=5)
            self.proc = None

    def tool(self, name: str, arguments: dict) -> dict:
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


_mcp: Optional[MCPClient] = None


def get_mcp() -> MCPClient:
    global _mcp
    if _mcp is None:
        _mcp = MCPClient()
        _mcp.start()
    return _mcp


# =============================================================================
# Turtle validation
# =============================================================================

def validate_turtle(turtle_str: str) -> tuple[bool, str]:
    """Check if Turtle is syntactically valid. Returns (ok, error_msg)."""
    try:
        from rdflib import Graph
        g = Graph()
        g.parse(data=turtle_str, format="turtle")
        triple_count = len(g)
        if triple_count == 0:
            return False, "Turtle parsed but contains 0 triples"
        return True, f"{triple_count} triples"
    except ImportError:
        # rdflib not installed — do basic checks
        if "@prefix" not in turtle_str and "PREFIX" not in turtle_str:
            return False, "No @prefix declarations found"
        if "arg:" not in turtle_str:
            return False, "No arg: namespace nodes found"
        # Count lines that look like triples
        triple_lines = [l for l in turtle_str.split("\n")
                       if l.strip() and not l.strip().startswith("@") and not l.strip().startswith("#")]
        if len(triple_lines) < 3:
            return False, f"Only {len(triple_lines)} triple-like lines"
        return True, f"~{len(triple_lines)} lines (rdflib not available for full validation)"
    except Exception as e:
        return False, f"Parse error: {e}"


def extract_turtle_from_text(text: str) -> Optional[str]:
    """Extract OWL Turtle from LLM output — handles code blocks, mixed text, etc."""
    # Try ```turtle ... ``` block first
    match = re.search(r"```(?:turtle|ttl|rdf)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        ok, _ = validate_turtle(candidate)
        if ok:
            return candidate

    # Try finding @prefix ... to the last triple
    match = re.search(r"(@prefix.*?)(?:\n\n[^@\s]|\Z)", text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        ok, _ = validate_turtle(candidate)
        if ok:
            return candidate

    # Last resort — if text itself has prefixes
    if "@prefix arg:" in text:
        ok, _ = validate_turtle(text)
        if ok:
            return text

    return None


# =============================================================================
# Direct scoring (no tool-calling, just prompt → parse JSON)
# =============================================================================

def direct_score(llm_cfg: dict, doc_text: str, criteria: list, agent_id: str, mcp: MCPClient) -> int:
    """Score directly by prompting Qwen and parsing JSON — no tool-calling needed."""
    criteria_list = "\n".join(f"- {c['id']}: {c['title']}" for c in criteria)

    prompt = f"""Score this document against each criterion on a 0-10 scale.
Return ONLY a JSON array, nothing else:
[{{"criterion_id": "...", "score": N, "justification": "..."}}]

Criteria:
{criteria_list}

Document (first 6000 chars):
{doc_text[:6000]}"""

    bot = Assistant(llm=llm_cfg, system_message="You are a document evaluator. Return only valid JSON.")
    messages = [{"role": "user", "content": prompt}]

    result_text = ""
    for response in bot.run(messages=messages):
        if isinstance(response, list):
            for msg in response:
                result_text = msg.get("content", "")

    # Parse JSON from response
    scores = []
    try:
        # Try direct parse
        scores = json.loads(result_text)
    except json.JSONDecodeError:
        # Try extracting JSON array from text
        match = re.search(r"\[.*\]", result_text, re.DOTALL)
        if match:
            try:
                scores = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    recorded = 0
    for s in scores:
        cid = s.get("criterion_id", "")
        score = s.get("score", 0)
        justification = s.get("justification", "")
        if cid and isinstance(score, (int, float)):
            r = mcp.tool("eval_record_score", {
                "agent_id": agent_id,
                "criterion_id": cid,
                "score": float(score),
                "max_score": 10.0,
                "justification": justification,
                "round": 1,
                "evidence_used": [],
                "gaps_identified": [],
            })
            if r.get("ok"):
                recorded += 1
                print(f"   {cid}: {score}/10")

    return recorded


# =============================================================================
# BITF Tools — registered with qwen-agent
# =============================================================================

@register_tool('bitf_ingest')
class BITFIngest(BaseTool):
    description = 'Ingest a document into the BITF evaluation server.'
    parameters = {
        'type': 'object',
        'properties': {
            'path': {'type': 'string', 'description': 'Path to the document'},
            'intent': {'type': 'string', 'description': 'Evaluation intent'},
        },
        'required': ['path', 'intent']
    }

    def call(self, params: str, **kwargs) -> str:
        return json.dumps(get_mcp().tool("eval_ingest", json.loads(params)), ensure_ascii=False)


@register_tool('bitf_criteria')
class BITFCriteria(BaseTool):
    description = 'Load evaluation criteria framework.'
    parameters = {
        'type': 'object',
        'properties': {
            'framework': {'type': 'string', 'description': 'Framework: generic, academic, tender, clinical'},
        },
        'required': ['framework']
    }

    def call(self, params: str, **kwargs) -> str:
        return json.dumps(get_mcp().tool("eval_criteria", json.loads(params)), ensure_ascii=False)


@register_tool('bitf_spawn')
class BITFSpawn(BaseTool):
    description = 'Spawn evaluator agent panel.'
    parameters = {
        'type': 'object',
        'properties': {
            'intent': {'type': 'string', 'description': 'Evaluation intent'},
        },
        'required': ['intent']
    }

    def call(self, params: str, **kwargs) -> str:
        return json.dumps(get_mcp().tool("eval_spawn", json.loads(params)), ensure_ascii=False)


@register_tool('bitf_load_turtle')
class BITFLoadTurtle(BaseTool):
    description = 'Load OWL Turtle into the GraphStore.'
    parameters = {
        'type': 'object',
        'properties': {
            'turtle': {'type': 'string', 'description': 'OWL Turtle string'},
        },
        'required': ['turtle']
    }

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        # Validate before loading
        ok, msg = validate_turtle(args.get("turtle", ""))
        if not ok:
            return json.dumps({"error": f"Invalid Turtle: {msg}"})
        return json.dumps(get_mcp().tool("eds_load_turtle", args), ensure_ascii=False)


@register_tool('bitf_record_score')
class BITFRecordScore(BaseTool):
    description = 'Record a score for a criterion.'
    parameters = {
        'type': 'object',
        'properties': {
            'agent_id': {'type': 'string'},
            'criterion_id': {'type': 'string'},
            'score': {'type': 'number'},
            'max_score': {'type': 'number'},
            'justification': {'type': 'string'},
        },
        'required': ['agent_id', 'criterion_id', 'score', 'max_score', 'justification']
    }

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        args.setdefault("round", 1)
        args.setdefault("evidence_used", [])
        args.setdefault("gaps_identified", [])
        return json.dumps(get_mcp().tool("eval_record_score", args), ensure_ascii=False)


@register_tool('bitf_score')
class BITFScore(BaseTool):
    description = 'Get the BITF verdict: SPARQL rules, structural score, gate.'
    parameters = {
        'type': 'object',
        'properties': {
            'agent_id': {'type': 'string'},
        },
        'required': ['agent_id']
    }

    def call(self, params: str, **kwargs) -> str:
        return json.dumps(get_mcp().tool("eds_score", json.loads(params)), ensure_ascii=False)


# =============================================================================
# Agent prompts
# =============================================================================

DECOMPOSER_PROMPT = """You are the Document Decomposer for the Hangzhou Tunneling pipeline.

Your job: read a document and output an OWL Turtle argument ontology.

IMPORTANT:
- Do NOT score nodes. Only identify, type, connect, and quote them.
- Every arg:hasText MUST be an EXACT copy-paste from the document. Do not paraphrase.
- Output ONLY the Turtle inside a ```turtle code block. No other text.

Node types: arg:Thesis, arg:SubClaim, arg:Evidence, arg:QuantifiedEvidence, arg:Citation, arg:Counter, arg:Rebuttal, arg:Structural

Relationships: arg:supports, arg:counters, arg:rebuts, arg:contains

Required prefixes:
@prefix arg: <http://brain-in-the-fish.dev/arg/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

Example node:
arg:s0_0 a arg:Thesis ;
    arg:hasText "Score any document. Prove every claim." .
arg:s0_1 a arg:QuantifiedEvidence ;
    arg:hasText "0 fabricated evidence out of 1,271 nodes tested" ;
    arg:supports arg:s0_0 ."""

SCORER_PROMPT = """You are the Document Scorer for the Hangzhou Tunneling pipeline.

Score the document against each criterion on a 0-10 scale.

Return ONLY a JSON array:
[{"criterion_id": "...", "score": N, "justification": "specific quote or reference"}]

Be rigorous. Reference specific content from the document."""

VERIFIER_PROMPT = """You are the Document Verifier for the Hangzhou Tunneling pipeline.

Call the bitf_score tool with the provided agent_id to get the BITF verdict.
Report the structural score, mined facts, and verdict."""


# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline(document_path: str, intent: str, llm_cfg: dict):
    """Run the full Hangzhou Tunneling pipeline with safeguards."""

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  HANGZHOU TUNNELING — Qwen Agents + BITF Verification      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if not os.path.exists(document_path):
        print(f"Error: {document_path} not found")
        sys.exit(1)

    doc_text = open(document_path).read()

    # Start MCP server
    print("\nStarting BITF MCP server...")
    try:
        mcp = get_mcp()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # ── Phase 1: Ingest + Decompose ───────────────────────────────────
    print("\n── Phase 1: Ingest & Decompose ─────────────────────────────")

    r = mcp.tool("eval_ingest", {"path": os.path.abspath(document_path), "intent": intent})
    if "error" in r:
        print(f"   Ingest failed: {r['error']}")
        mcp.stop()
        sys.exit(1)
    print(f"   Ingested: {r.get('sections', '?')} sections, {r.get('triples_loaded', '?')} triples")

    r = mcp.tool("eval_criteria", {"framework": "generic"})
    criteria = r.get("criteria", [])
    print(f"   Criteria: {len(criteria)} loaded")

    r = mcp.tool("eval_spawn", {"intent": intent})
    agents = r.get("agents", [])
    agent_id = agents[0]["id"] if agents else None
    if not agent_id:
        print("   Spawn failed — no agents")
        mcp.stop()
        sys.exit(1)
    print(f"   Agents: {len(agents)} spawned, primary: {agent_id}")

    # Run Qwen Decomposer
    print("\n   Running Qwen Decomposer...")
    decomposer = Assistant(
        llm=llm_cfg,
        name='Decomposer',
        system_message=DECOMPOSER_PROMPT,
    )
    decompose_msg = [{"role": "user", "content": f"Decompose this document into OWL Turtle. Use EXACT quotes only.\n\n{doc_text[:8000]}"}]

    turtle_str = None
    try:
        for response in decomposer.run(messages=decompose_msg):
            if isinstance(response, list):
                for msg in response:
                    content = msg.get("content", "")
                    if content:
                        extracted = extract_turtle_from_text(content)
                        if extracted:
                            turtle_str = extracted
    except Exception as e:
        print(f"   Decomposer error: {e}")

    # Validate and load Turtle
    turtle_loaded = False
    if turtle_str:
        ok, msg = validate_turtle(turtle_str)
        if ok:
            r = mcp.tool("eds_load_turtle", {"turtle": turtle_str})
            node_count = r.get("node_count", 0)
            if node_count > 0:
                print(f"   Turtle loaded: {node_count} nodes, {r.get('triples_loaded', '?')} triples ({msg})")
                turtle_loaded = True
            else:
                print(f"   Turtle loaded but 0 nodes detected — falling back")
        else:
            print(f"   Invalid Turtle: {msg} — falling back")
    else:
        print(f"   Decomposer did not produce Turtle — falling back")

    if not turtle_loaded:
        print("   Fallback: using deterministic ingest (no ontology decomposition)")
        # The MCP server already has the document from eval_ingest —
        # structural score will be based on ingest triples only

    # ── Phase 2: Score ────────────────────────────────────────────────
    print("\n── Phase 2: Score ──────────────────────────────────────────")

    # Try agent-based tool-calling first
    scorer = Assistant(
        llm=llm_cfg,
        name='Scorer',
        system_message=SCORER_PROMPT,
        function_list=['bitf_record_score']
    )
    criteria_info = json.dumps({c["title"]: c["id"] for c in criteria})
    score_msg = [{"role": "user", "content": (
        f"Score this document. agent_id={agent_id}\n"
        f"Criterion IDs: {criteria_info}\n"
        f"max_score=10 for all.\n\n"
        f"Document:\n{doc_text[:6000]}"
    )}]

    tool_scores_recorded = 0
    try:
        for response in scorer.run(messages=score_msg):
            if isinstance(response, list):
                for msg in response:
                    content = msg.get("content", "")
                    if "bitf_record_score" in str(msg.get("function_call", "")):
                        tool_scores_recorded += 1
    except Exception as e:
        print(f"   Scorer agent error: {e}")

    if tool_scores_recorded > 0:
        print(f"   Agent recorded {tool_scores_recorded} scores via tool-calling")
    else:
        # Fallback: direct scoring (prompt → JSON → manual record)
        print("   Agent tool-calling failed — using direct scoring fallback")
        recorded = direct_score(llm_cfg, doc_text, criteria, agent_id, mcp)
        print(f"   Direct scoring recorded {recorded}/{len(criteria)} scores")

    # ── Phase 3: Verify ───────────────────────────────────────────────
    print("\n── Phase 3: Verify ─────────────────────────────────────────")

    # Always call directly — don't rely on agent tool-calling for the verdict
    r = mcp.tool("eds_score", {"agent_id": agent_id})

    structural = r.get("structural_score", "?")
    topo = r.get("topology", {})
    mined = r.get("mined_facts", {})
    verdict = r.get("verdict", "")

    print(f"   Structural score: {structural}")
    if topo:
        print(f"   Nodes: {topo.get('node_count', '?')} | Evidence: {topo.get('evidence_count', '?')} | Claims: {topo.get('claim_count', '?')}")
    if mined:
        print(f"   SPARQL rules: strong={mined.get('strong_claims', '?')} weak={mined.get('weak_claims', '?')} unsupported={mined.get('unsupported_claims', '?')}")

    # Audit trail
    audit = r.get("audit_trail", [])
    if audit:
        print(f"   Audit trail: {len(audit)} nodes")

    verdict_str = str(verdict)
    if "CONFIRMED" in verdict_str:
        badge = "VERIFIED"
    elif "FLAGGED" in verdict_str:
        badge = "FLAGGED"
    else:
        badge = "REJECTED"

    print(f"\n   ╔═══════════════════════════════════════════════════╗")
    print(f"   ║  VERDICT: {verdict_str[:48]}")
    print(f"   ╚═══════════════════════════════════════════════════╝")

    mcp.stop()

    print(f"\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║  PIPELINE COMPLETE — Badge: BITF {badge:<25}║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    return badge


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "evaluate":
        print("Usage: python hangzhou_tunneling.py evaluate <document> --intent <intent>")
        print()
        print("Options:")
        print("  --intent <text>    What to evaluate (required)")
        print("  --model <name>     Qwen model (default: Qwen3-8B)")
        print("  --server <url>     Model server URL (default: http://localhost:11434/v1)")
        print("  --dashscope        Use DashScope API instead of local server")
        sys.exit(1)

    document = sys.argv[2]
    intent = "assess document quality"
    model = "Qwen3-8B"
    server_url = "http://localhost:11434/v1"
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
            "model": model if model != "Qwen3-8B" else "qwen-max-latest",
            "model_type": "qwen_dashscope",
            "generate_cfg": {"top_p": 0.8},
        }
    else:
        llm_cfg = {
            "model": model,
            "model_server": server_url,
            "api_key": "EMPTY",
            "generate_cfg": {"top_p": 0.8},
        }

    run_pipeline(document, intent, llm_cfg)
