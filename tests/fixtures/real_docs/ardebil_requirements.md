# Project Requirements: Multi-Intent RAG Chatbot v2.0

## Overview
Upgrade the existing chatbot (v1.0 with Intent Detection, Slot Filling, and Dual-Source RAG) to include self-evaluation and API deployment capabilities.

---

## Part 1: Self-Evaluation & Clarification Layer

<requirement id="1.1" priority="critical">
  <title>Response Self-Evaluation</title>
  <description>
    After generating a response, the model MUST evaluate its own answer across three dimensions:
  </description>
  <metrics>
    <metric name="groundedness">
      <definition>Is the response based on the retrieved source/context?</definition>
      <range>0.0 to 1.0</range>
    </metric>
    <metric name="coherence">
      <definition>Is the response coherent and well-structured?</definition>
      <range>0.0 to 1.0</range>
    </metric>
    <metric name="confidence">
      <definition>Model's confidence in the final response</definition>
      <range>0.0 to 1.0</range>
    </metric>
  </metrics>
  <output_format>JSON</output_format>
</requirement>

<requirement id="1.2" priority="critical">
  <title>Clarifying Questions for Low Confidence</title>
  <description>
    When confidence &lt; threshold (default: 0.65), the system MUST NOT return an uncertain answer.
    Instead, it should:
  </description>
  <actions>
    <action>Ask 2-3 clarifying questions (e.g., "Do you mean product availability or compatibility?")</action>
    <action>OR rewrite the user's request to remove ambiguity</action>
  </actions>
  <example>
    User: "Is it good?"
    System: "I need more context. Are you asking about:
    1. Product quality?
    2. Price value?
    3. Compatibility with other products?"
  </example>
</requirement>

<requirement id="1.3" priority="high">
  <title>Standalone Evaluation Function</title>
  <description>
    The evaluate_answer() module MUST be callable independently with a separate evaluation prompt.
  </description>
  <signature>
    evaluate_answer(query: str, context: List[str], response: str) → EvaluationMetrics
  </signature>
  <use_cases>
    <case>Testing and debugging responses</case>
    <case>Batch evaluation of historical queries</case>
    <case>Integration with external systems</case>
  </use_cases>
</requirement>

---

## Part 2: API Deployment (FastAPI + Docker)

<requirement id="2.1" priority="critical">
  <title>FastAPI Endpoints</title>
  <description>Create a FastAPI application with at least three endpoints:</description>

  <endpoint method="POST" path="/chat">
    <input>
      <field name="query" type="string" required="true">User's question</field>
      <field name="confidence_threshold" type="float" required="false" default="0.65">Minimum confidence threshold</field>
    </input>
    <output>
      <field name="query">Original user query</field>
      <field name="intent">Detected intent (Product QA / FAQ)</field>
      <field name="slots">Extracted slot values</field>
      <field name="response">Generated answer</field>
      <field name="evaluation">
        <subfield name="groundedness">Score 0-1</subfield>
        <subfield name="coherence">Score 0-1</subfield>
        <subfield name="confidence">Score 0-1</subfield>
        <subfield name="reasoning">Explanation</subfield>
      </field>
      <field name="clarifying_questions">List of questions (if confidence low) or null</field>
    </output>
  </endpoint>

  <endpoint method="GET" path="/metrics">
    <description>Return aggregated system metrics</description>
    <output>
      <field name="total_queries">Total number of processed queries</field>
      <field name="avg_confidence">Average confidence score</field>
      <field name="clarify_rate">Percentage of queries needing clarification</field>
      <field name="avg_latency_ms">Average response time</field>
      <field name="intent_distribution">Query count per intent type</field>
    </output>
  </endpoint>

  <endpoint method="POST" path="/evaluate">
    <description>Standalone evaluation endpoint (optional: can run automated tests and return summary)</description>
    <input>
      <field name="query" type="string">User query</field>
      <field name="context" type="list">Retrieved context chunks</field>
      <field name="response" type="string">Model's response</field>
    </input>
    <output>
      <field name="groundedness">Score 0-1</field>
      <field name="coherence">Score 0-1</field>
      <field name="confidence">Score 0-1</field>
      <field name="reasoning">Detailed explanation</field>
    </output>
  </endpoint>
</requirement>

<requirement id="2.2" priority="critical">
  <title>Structured Logging</title>
  <description>
    Implement structured logging in JSON format for every request.
  </description>
  <format>JSON Lines (JSONL)</format>
  <required_fields>
    <field>timestamp</field>
    <field>level (INFO/ERROR/WARNING)</field>
    <field>request_id</field>
    <field>query</field>
    <field>intent</field>
    <field>confidence</field>
    <field>latency_ms</field>
    <field>error (if any)</field>
  </required_fields>
  <storage>
    <requirement>Logs MUST be human-readable</requirement>
    <requirement>Metrics MUST be easily visible, not abstract</requirement>
    <note>User prefers human-readable formats over binary databases</note>
  </storage>
</requirement>

---

## Technical Constraints

<constraint type="model">
  <primary_model>gemini-flash-latest</primary_model>
  <usage>
    <chatbot>Use gemini-flash-latest for response generation</chatbot>
    <evaluator>Use gemini-flash-latest for evaluation (LLM as judge)</evaluator>
  </usage>
  <note>Ollama is acceptable but slow; Gemini preferred for speed</note>
</constraint>

<constraint type="validation">
  <approach>LLM as judge with Pydantic-style validation loop</approach>
  <description>
    If the model generates a wrong answer, the evaluator should prompt it to fix its mistake.
    Similar to Pydantic validation: keep regenerating until the response passes validation criteria.
  </description>
  <retry_logic>
    <max_retries>TBD (clarify with PM)</max_retries>
    <feedback_format>Provide specific error feedback to model for regeneration</feedback_format>
  </retry_logic>
</constraint>

---

## Success Criteria

<acceptance_criteria>
  <criterion id="AC-1">Every response includes groundedness, coherence, and confidence scores</criterion>
  <criterion id="AC-2">Low-confidence responses trigger clarifying questions instead of uncertain answers</criterion>
  <criterion id="AC-3">All three API endpoints functional and return correct JSON structure</criterion>
  <criterion id="AC-4">Structured logs are human-readable and metrics easily accessible</criterion>
  <criterion id="AC-5">evaluate_answer() callable independently via Python and REST API</criterion>
  <criterion id="AC-6">Comprehensive test case documentation showing actual evaluation results</criterion>
</acceptance_criteria>

---

## Open Questions (Require PM Clarification)

<question id="Q1">
  <topic>Validation Loop Retry Logic</topic>
  <details>
    - Max retry attempts for regeneration?
    - What happens if answer never passes validation?
    - Return best attempt or explicit failure?
  </details>
</question>

<question id="Q2">
  <topic>Human-Readable Metrics Format</topic>
  <details>
    - Preferred format: CSV, JSON files, or plain text logs?
    - Should metrics be queryable or just readable?
  </details>
</question>

<question id="Q3">
  <topic>Validation Failure Criteria</topic>
  <details>
    - What thresholds determine a "wrong" answer needing regeneration?
    - Groundedness &lt; X? Coherence &lt; Y? Both?
  </details>
</question>
