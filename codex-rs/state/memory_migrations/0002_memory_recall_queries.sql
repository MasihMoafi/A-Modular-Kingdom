CREATE TABLE stage1_recall_queries (
    thread_id TEXT NOT NULL,
    query_key TEXT NOT NULL,
    recalled_at INTEGER NOT NULL,
    PRIMARY KEY (thread_id, query_key),
    FOREIGN KEY (thread_id) REFERENCES stage1_outputs(thread_id) ON DELETE CASCADE
);

CREATE INDEX idx_stage1_recall_queries_thread_id
ON stage1_recall_queries(thread_id);
