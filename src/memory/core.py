# mem0_memory.py - Qdrant-based memory system
import os

# Central proxy manager — strip all proxy for local Qdrant
from utils.proxy import strip_all as _strip_proxy
_strip_proxy()

import uuid
import json
import re
from typing import List, Dict, Optional
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi

from memory.memory_config import MemoryConfig as _MemoryConfig

LLM_MODEL = 'qwen3:8b'
COLLECTION_NAME = _MemoryConfig.DEFAULT_COLLECTION_NAME
EMBED_MODEL = _MemoryConfig.DEFAULT_EMBED_MODEL

def log_message(message: str):
    """Disabled debug logging."""
    pass

class Mem0:
    def __init__(self, storage_path: str, collection_name: str = COLLECTION_NAME):
        """Initialize Mem0 memory system with Qdrant vector database.

        Args:
            storage_path: Path for Qdrant storage
            collection_name: Name of collection
        """
        self.storage_path = storage_path
        self.collection_name = collection_name

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

        # Qdrant client (local mode)
        self._client = QdrantClient(path=str(Path(self.storage_path) / "qdrant_storage"))

        # In-memory BM25 index
        self._bm25_docs: List[List[str]] = []
        self._bm25_ids: List[str] = []
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_dirty: bool = True

        self._init_collection()
        self._vector_size = self._get_collection_vector_size()

    def _get_collection_vector_size(self) -> int:
        """Best-effort: read vector size from Qdrant collection config."""
        try:
            info = self._client.get_collection(collection_name=self.collection_name)
            # qdrant-client returns a strongly-typed object; these attributes exist in practice.
            return int(info.config.params.vectors.size)  # type: ignore[attr-defined]
        except Exception:
            return 768  # embeddinggemma default

    def _safe_embedding(self, text: str) -> List[float]:
        """Return embedding or a zero-vector fallback when Ollama is unavailable.

        This keeps memory tools functional (BM25-based) even if Ollama isn't running.
        """
        try:
            return self._get_embedding(text)
        except Exception:
            return [0.0] * int(getattr(self, "_vector_size", 768))

    def _init_collection(self):
        """Initialize Qdrant collection."""
        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            # Get embedding dimension from Ollama
            try:
                test_embed = self._get_embedding("test")
                dim = len(test_embed)
            except:
                dim = 768  # Default for embeddinggemma

            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from Ollama."""
        try:
            import ollama
            response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
            return response['embedding']
        except Exception as e:
            log_message(f"Embedding error: {e}")
            raise

    def _execute_operation(self, operation: str, fact: str, memory_id: str = None):
        """Execute memory operation (ADD/UPDATE)."""
        log_message(f"DEBUG: Executing operation: Op='{operation}', Fact='{fact}', ID='{memory_id}'")
        try:
            if operation == 'ADD':
                new_id = str(uuid.uuid4())
                vector = self._safe_embedding(fact)
                point_id = abs(hash(new_id)) % (2**63)

                self._client.upsert(
                    collection_name=self.collection_name,
                    points=[PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={"memory_id": new_id, "content": fact}
                    )]
                )
                log_message(f"[Mem0] HOST ACTION: ADDED new memory. ID: {new_id[:8]}")
                self._bm25_dirty = True

            elif operation == 'UPDATE' and memory_id:
                vector = self._safe_embedding(fact)
                point_id = abs(hash(memory_id)) % (2**63)

                self._client.upsert(
                    collection_name=self.collection_name,
                    points=[PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={"memory_id": memory_id, "content": fact}
                    )]
                )
                log_message(f"[Mem0] HOST ACTION: UPDATED memory. ID: {memory_id[:8]}")
                self._bm25_dirty = True

        except Exception as e:
            log_message(f"[Mem0] HOST ERROR during operation: {e}")

    def _decide_memory_operation(self, fact: str, similar_memories: List[Dict]) -> Dict:
        """Use LLM to decide if fact should be added, updated, or ignored."""
        if not similar_memories:
            log_message("[Mem0] No similar memories found. Defaulting to ADD.")
            return {"operation": "ADD", "fact": fact}

        similar_memories_str = "\n".join([
            f"- ID: {mem['id']}, Fact: {mem['content']}"
            for mem in similar_memories
        ])

        prompt = f"""You are a memory consolidation system. Your goal is to maintain a concise and accurate fact store.

Follow these rules:
1. A correction (e.g., 'no, it's a DSL button') should ALWAYS be an UPDATE.
2. If the "New Fact" is semantically identical or redundant to an "Existing Memory", you MUST use the "NOOP" operation. Do not add duplicate information.

Based on the 'New Fact', decide if it should be added as a new memory, update an existing one, or be ignored.

Existing Memories:
{similar_memories_str}

New Fact: "{fact}"

Your decision MUST be a single JSON object with either:
1. {{"operation": "ADD", "fact": "the new fact to add"}}
2. {{"operation": "UPDATE", "memory_id": "the id of the memory to update", "updated_fact": "the new, corrected fact"}}
3. {{"operation": "NOOP", "reason": "why no action is needed"}}

Decision:"""
        try:
            import ollama
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )
            raw_response_content = response['message']['content']
            log_message(f"DEBUG: Raw LLM Response:\n{raw_response_content}")

            decision = json.loads(raw_response_content)
            log_message(f"DEBUG: Parsed Decision: {json.dumps(decision, indent=2)}")
            return decision

        except (json.JSONDecodeError, KeyError, Exception) as e:
            log_message(f"HOST ERROR: Could not get valid decision from LLM: {e}")
            return {"operation": "ADD", "fact": fact}

    def _extract_facts(self, conversation: str) -> List[str]:
        """Extract facts from conversation using LLM."""
        log_message(f"DEBUG: Extracting facts from conversation")
        prompt = f"""You are a hyper-critical fact extractor. Your job is to extract only the most essential, enduring facts from a conversation.

Follow these rules STRICTLY:
1.  A fact MUST be a definitive statement of truth (e.g., "The user's car is red," "There is no DNS button on the modem").
2.  A fact MUST be a correction of a previous misunderstanding.
3.  A fact MUST be a core user preference (e.g., "The user is a vegetarian").
4.  IGNORE conversational filler, greetings, questions, or uncertainties (e.g., "I think...", "Maybe...", "Hello", "How are you?").
5.  The fact must be self-contained and understandable on its own.

Return a JSON object with a single key "facts" containing a list of strings. If no essential facts are found, return an empty list.

Conversation:
{conversation}

JSON Output:"""
        try:
            import ollama
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )
            raw_response_content = response['message']['content']
            log_message(f"DEBUG: Raw fact extraction: {raw_response_content}")
            data = json.loads(raw_response_content)
            return data.get("facts", [])
        except (json.JSONDecodeError, KeyError, Exception) as e:
            log_message(f"HOST ERROR: Could not extract facts: {e}")
            return []

    # --- BM25 helpers ---
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        text = (text or "").lower()
        return [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]

    def _rebuild_bm25(self):
        """Rebuild BM25 index from Qdrant."""
        try:
            results = self._client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
            )

            ids = []
            docs = []
            for point in results[0]:
                ids.append(point.payload.get("memory_id", ""))
                docs.append(point.payload.get("content", ""))

            tokenized = [self._tokenize(d) for d in docs]

            self._bm25_ids = ids
            self._bm25_docs = tokenized
            self._bm25 = BM25Okapi(tokenized) if tokenized else None
            self._bm25_dirty = False
        except Exception as e:
            log_message(f"BM25 rebuild error: {e}")
            self._bm25 = None
            self._bm25_dirty = True

    # --- Public API ---
    def direct_add(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Add content directly without LLM processing."""
        memory_id = str(uuid.uuid4())
        vector = self._safe_embedding(content)
        point_id = abs(hash(memory_id)) % (2**63)

        payload = {
            "memory_id": memory_id,
            "content": content,
            **(metadata or {})
        }

        self._client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)]
        )
        self._bm25_dirty = True
        return memory_id

    def direct_delete(self, memory_id: str) -> bool:
        """Delete memory by ID. Returns True if a memory was deleted."""
        try:
            if self._get_by_id(memory_id) is None:
                return False
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="memory_id", match=MatchValue(value=memory_id))]
                )
            )
            self._bm25_dirty = True
            return True
        except Exception as e:
            log_message(f"Delete error: {e}")
            return False

    def add(self, conversation: str):
        """Add conversation with LLM fact extraction and deduplication."""
        facts = self._extract_facts(conversation)
        log_message(f"[Mem0] Extracted {len(facts)} facts from conversation.")

        for fact in facts:
            similar = self.search(fact, k=3)
            decision = self._decide_memory_operation(fact, similar)
            operation = decision.get("operation", "ADD")

            if operation == "ADD":
                self._execute_operation("ADD", decision.get("fact", fact))
            elif operation == "UPDATE":
                memory_id = decision.get("memory_id")
                updated_fact = decision.get("updated_fact")
                if memory_id and updated_fact:
                    self._execute_operation("UPDATE", updated_fact, memory_id)
            elif operation == "NOOP":
                log_message(f"[Mem0] NOOP: {decision.get('reason', 'No reason given')}")

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search memories using BM25 (primary) with vector fallback."""
        try:
            if self._bm25_dirty or self._bm25 is None:
                self._rebuild_bm25()

            if self._bm25 is not None and self._bm25_docs:
                q_tokens = self._tokenize(query)
                scores = self._bm25.get_scores(q_tokens)
                ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

                results = []
                for idx, score in ranked[:k]:
                    if score > 0:
                        memory_id = self._bm25_ids[idx]
                        memory = self._get_by_id(memory_id)
                        if memory:
                            results.append(memory)

                if results:
                    return results
        except Exception as e:
            log_message(f"BM25 search error: {e}")

        # Fallback to vector search
        return self._vector_search(query, k)

    def _get_by_id(self, memory_id: str) -> Optional[Dict]:
        """Get memory by ID."""
        try:
            results = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="memory_id", match=MatchValue(value=memory_id))]
                ),
                limit=1,
                with_payload=True,
            )
            if results[0]:
                point = results[0][0]
                return {
                    "id": point.payload.get("memory_id"),
                    "content": point.payload.get("content", ""),
                }
        except:
            pass
        return None

    def _vector_search(self, query: str, k: int) -> List[Dict]:
        """Fallback vector similarity search."""
        try:
            query_vector = self._safe_embedding(query)
            results = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=k,
                with_payload=True,
            )

            return [{
                "id": hit.payload.get("memory_id"),
                "content": hit.payload.get("content", ""),
            } for hit in results.points]
        except:
            return []

    def get_all_memories(self) -> List[Dict]:
        """Retrieve all memories."""
        try:
            results = self._client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
            )

            return [{
                "id": point.payload.get("memory_id"),
                "content": point.payload.get("content", ""),
            } for point in results[0]]
        except:
            return []
