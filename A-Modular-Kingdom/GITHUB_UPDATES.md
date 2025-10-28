# GitHub Repository Updates Needed

## A-Modular-Kingdom (FLAGSHIP)
**Current**: No description
**New Description**: "Production-ready AI infrastructure: RAG with smart reindexing, persistent memory, browser automation, and MCP integration. Stop rebuilding tools for every AI project."

**Topics to Add**: 
`mcp` `rag` `ai-agents` `llm` `memory-system` `browser-automation` `ollama` `playwright` `python` `ai-infrastructure`

**README**: ✅ UPDATED (clearer value prop, demos, use cases)

---

## Eyes-Wide-Shut (5⭐)
**Current**: No description
**New Description**: "LLM security research: Red-teaming GPT-OSS-20B. Discovered 5 high-severity vulnerabilities including cross-lingual attacks and semantic exploits."

**Topics to Add**:
`llm-security` `red-teaming` `ai-safety` `vulnerability-research` `prompt-injection` `python`

**README**: Already good (detailed write-up exists)

---

## Voice-commander (5⭐)
**Current**: "A local transcriber wherever you go."
**Keep**: Good as-is

**Topics to Add**:
`whisper` `speech-to-text` `voice-transcription` `gpu-acceleration` `gemini-api` `python` `productivity`

**README**: Already excellent (demo video, clear features)

---

## google-Hackathon (0⭐)
**Current**: "Hackathon hosted by google on kaggle to make impactful use of gemma3n."
**New Description**: "Google Kaggle Hackathon submission: Multi-agent emotional AI using Gemma 3n's multimodal capabilities (voice, vision) with RAG and persistent memory."

**Topics to Add**:
`gemma` `kaggle` `hackathon` `multimodal-ai` `emotion-detection` `rag` `python`

**README**: Already good (detailed architecture)

---

## Financial-Market-Analysis (1⭐)
**Current**: "Time-Series Analysis, Clustering, Transformers, Rocket"
**New Description**: "Financial market analysis using time-series models, clustering algorithms, Transformers, and reinforcement learning for trading strategies."

**Topics to Add**:
`time-series` `financial-analysis` `transformers` `reinforcement-learning` `lstm` `clustering` `jupyter-notebook`

**README**: Needs improvement (add results, findings, visualizations)

---

## OpenCV-Tutorial (4⭐)
**Current**: "OpenCV-Tutorial for Computer Vision course; Oct, 2025."
**Keep**: Good as-is

**Topics to Add**:
`opencv` `computer-vision` `tutorial` `image-processing` `python` `education`

**README**: Check if complete

---

## ML-Exercises (0⭐)
**Current**: "ML Exercises Oct, 2025."
**New Description**: "Machine Learning exercises and implementations from university coursework (Oct 2025). Covers supervised/unsupervised learning, neural networks, and optimization."

**Topics to Add**:
`machine-learning` `coursework` `neural-networks` `python` `education`

**README**: Needs improvement (add topics covered, key implementations)

---

## ARCHIVE/PRIVATE

### My_freelance_website
**Action**: Make PRIVATE (not deleted)

### masihmoafi.github.io
**Action**: Archive if not using, or update if it's your portfolio

### claude-code-templates-A-Modular-Kingdom (Fork)
**Action**: Delete (forked repo, no value add)

### Using-Pydantic-For-Structured-Outputs
**Action**: Archive (tutorial-style, not a project)

### Cross-modal-knowledge-guided-model-for-AS
**Action**: Archive or complete (incomplete name, unclear purpose)

---

## GitHub Profile README Updates

**Current Issues**:
- Too many sections (cluttered)
- Generic badges
- No clear focus

**Recommendations**:
1. Lead with A-Modular-Kingdom (flagship project)
2. Remove generic badges, keep only meaningful stats
3. Simplify sections:
   - 🏆 Featured Projects (3 max)
   - 🔬 Research (Eyes-Wide-Shut)
   - 📚 Learning (OpenCV, ML-Exercises)
4. Add "Currently working on" section
5. Keep tech stack minimal and relevant

---

## Priority Actions (Do These First)

1. ✅ Update A-Modular-Kingdom README (DONE)
2. Add description to A-Modular-Kingdom repo
3. Add topics/tags to all repos
4. Make My_freelance_website private
5. Archive/delete low-value repos
6. Update Financial-Market-Analysis README with results
7. Simplify GitHub profile README

---

## Commands to Run (via GitHub CLI)

```bash
# Update A-Modular-Kingdom description
gh repo edit MasihMoafi/A-Modular-Kingdom --description "Production-ready AI infrastructure: RAG with smart reindexing, persistent memory, browser automation, and MCP integration"

# Add topics
gh repo edit MasihMoafi/A-Modular-Kingdom --add-topic mcp,rag,ai-agents,llm,memory-system,browser-automation,ollama,playwright,python,ai-infrastructure

# Update Eyes-Wide-Shut
gh repo edit MasihMoafi/Eyes-Wide-Shut --description "LLM security research: Red-teaming GPT-OSS-20B. Discovered 5 high-severity vulnerabilities"
gh repo edit MasihMoafi/Eyes-Wide-Shut --add-topic llm-security,red-teaming,ai-safety,vulnerability-research,prompt-injection,python

# Make private
gh repo edit MasihMoafi/My_freelance_website --visibility private

# Archive repos
gh repo archive MasihMoafi/masihmoafi.github.io
gh repo archive MasihMoafi/Using-Pydantic-For-Structured-Outputs
gh repo archive MasihMoafi/Cross-modal-knowledge-guided-model-for-AS

# Delete fork
gh repo delete MasihMoafi/claude-code-templates-A-Modular-Kingdom --yes
```
