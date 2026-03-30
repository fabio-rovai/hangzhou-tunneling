# Hangzhou Tunneling

Qwen-Agent powered document evaluation with [Brain in the Fish](https://github.com/fabio-rovai/brain-in-the-fish) verification.

Three Qwen agents orchestrate document evaluation. The Rust engine verifies everything.

```
Qwen Decomposer → builds OWL Turtle from document
Qwen Scorer     → scores each criterion
Qwen Verifier   → calls BITF gate for final verdict
        ↓
   BITF Rust Engine (GraphStore → SPARQL rules → structural score → gate)
```

## Install

```bash
# 1. Install Qwen-Agent
pip install -U "qwen-agent[mcp]"

# 2. Build BITF (needs Rust)
git clone https://github.com/fabio-rovai/brain-in-the-fish
cd brain-in-the-fish
cargo build --release
export BITF_SERVER="$PWD/target/release/brain-in-the-fish-mcp"

# 3. Run with local Qwen model (via Ollama or vLLM)
python hangzhou_tunneling.py evaluate document.pdf --intent "assess methodology"

# Or with DashScope API
python hangzhou_tunneling.py evaluate document.pdf --intent "assess methodology" --dashscope
```

## How it works

| Phase | Agent | What happens |
|-------|-------|-------------|
| 1. Decompose | Qwen Decomposer | Reads document, builds OWL Turtle with typed nodes and exact quotes |
| 2. Score | Qwen Scorer | Scores each criterion 0-10 with justification |
| 3. Verify | Qwen Verifier | Calls BITF engine: SPARQL rules, topology score, gate verdict |

The Qwen agents do the thinking. The Rust engine does the verification. No Claude dependency.

## Model options

```bash
# Local (Ollama)
ollama serve
ollama pull qwen3:8b
python hangzhou_tunneling.py evaluate doc.pdf --intent "..." --server http://localhost:11434/v1 --model qwen3:8b

# Local (vLLM)
vllm serve Qwen/Qwen3-8B
python hangzhou_tunneling.py evaluate doc.pdf --intent "..."

# DashScope API
export DASHSCOPE_API_KEY=your_key
python hangzhou_tunneling.py evaluate doc.pdf --intent "..." --dashscope
```

## Architecture

```
hangzhou_tunneling.py
├── MCPClient          — JSON-RPC connection to BITF Rust server
├── BITF Tools         — 6 tools registered with qwen-agent
│   ├── bitf_ingest
│   ├── bitf_criteria
│   ├── bitf_spawn
│   ├── bitf_load_turtle
│   ├── bitf_record_score
│   └── bitf_score
└── Agents
    ├── Decomposer     — builds OWL Turtle (no scores, structure only)
    ├── Scorer         — scores criteria with justification
    └── Verifier       — gets gate verdict from Rust engine
```

## License

MIT
