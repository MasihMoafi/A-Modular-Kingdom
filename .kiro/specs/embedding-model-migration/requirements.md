# Requirements Document

## Introduction

This feature migrates the A-Modular-Kingdom MCP host from using SentenceTransformer embeddings (`all-MiniLM-L6-v2`) to Ollama's `embeddinggemma` model. The migration must maintain compatibility with existing functionality while eliminating the need to download separate embedding models, since Ollama is already being used for LLM operations.

## Requirements

### Requirement 1: Embedding Model Migration

**User Story:** As a developer, I want to use Ollama's embeddinggemma model for embeddings, so that I have a unified embedding solution without needing separate SentenceTransformer models.

#### Acceptance Criteria

1. WHEN the RAG system initializes THEN it SHALL use Ollama's embeddinggemma model instead of SentenceTransformer
2. WHEN embeddings are generated THEN they SHALL be compatible with the existing VectorIndex implementation
3. WHEN the memory system initializes THEN it SHALL use Ollama's embeddinggemma model for ChromaDB embeddings
4. IF Ollama is not available THEN the system SHALL provide a clear error message

### Requirement 2: Backward Compatibility

**User Story:** As a developer, I want existing indexed documents to be re-indexed automatically, so that I don't lose my knowledge base during the migration.

#### Acceptance Criteria

1. WHEN the system detects a model change THEN it SHALL trigger automatic re-indexing
2. WHEN re-indexing occurs THEN it SHALL preserve document metadata and structure
3. WHEN the migration completes THEN all existing functionality SHALL work without modification

### Requirement 3: Configuration Management

**User Story:** As a developer, I want the embedding model to be configurable, so that I can easily switch models if needed.

#### Acceptance Criteria

1. WHEN the RAG config is loaded THEN it SHALL specify the embedding model type and name
2. WHEN the memory system initializes THEN it SHALL use the same embedding configuration
3. WHEN the embedding model changes THEN the system SHALL detect and handle the change appropriately

### Requirement 4: MCP Integration

**User Story:** As a user of Claude, Codex, Gemini, or other MCP clients, I want to connect to the host.py server seamlessly, so that I can access all 14 tools without configuration issues.

#### Acceptance Criteria

1. WHEN an MCP client connects THEN it SHALL have access to all memory, RAG, and tool functions
2. WHEN embeddings are generated THEN they SHALL not cause connection timeouts or hangs
3. WHEN the host runs THEN it SHALL be stable and responsive to all MCP client requests
