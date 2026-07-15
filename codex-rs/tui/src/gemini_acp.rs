use std::path::Path;
use std::process::Stdio;

use serde_json::Value;
use serde_json::json;
use thiserror::Error;
use tokio::io::AsyncBufRead;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWrite;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::io::Lines;
use tokio::process::Child;
use tokio::process::ChildStdin;
use tokio::process::ChildStdout;
use tokio::process::Command;

const ACP_PROTOCOL_VERSION: u64 = 1;

#[derive(Debug, Error)]
pub enum GeminiAcpError {
    #[error("failed to start Gemini CLI: {0}")]
    Launch(#[source] std::io::Error),

    #[error("Gemini CLI did not expose its {0}")]
    MissingPipe(&'static str),

    #[error("Gemini ACP transport failed: {0}")]
    Io(#[from] std::io::Error),

    #[error("Gemini ACP returned malformed JSON: {0}")]
    MalformedJson(#[from] serde_json::Error),

    #[error("Gemini ACP closed before replying")]
    Closed,

    #[error("Gemini ACP returned response id {actual}, expected {expected}")]
    UnexpectedResponseId { expected: u64, actual: Value },

    #[error("Gemini ACP error {code}: {message}")]
    Rpc { code: i64, message: String },

    #[error("Gemini ACP response is missing {0}")]
    MissingField(&'static str),

    #[error("Gemini ACP protocol version {actual} is unsupported; expected {expected}")]
    UnsupportedProtocol { expected: u64, actual: u64 },

    #[error("Gemini ACP does not support loading sessions")]
    MissingLoadSession,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeminiAcpInitialize {
    pub agent_name: String,
    pub agent_version: String,
    pub auth_method_ids: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeminiAcpSession {
    pub session_id: String,
}

pub struct GeminiAcpClient {
    child: Child,
    connection: AcpConnection<BufReader<ChildStdout>, ChildStdin>,
}

impl GeminiAcpClient {
    pub fn launch(executable: &str, cwd: &Path) -> Result<Self, GeminiAcpError> {
        let mut child = Command::new(executable)
            .arg("--acp")
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(GeminiAcpError::Launch)?;

        let stdin = child
            .stdin
            .take()
            .ok_or(GeminiAcpError::MissingPipe("standard input"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or(GeminiAcpError::MissingPipe("standard output"))?;

        Ok(Self {
            child,
            connection: AcpConnection::new(BufReader::new(stdout), stdin),
        })
    }

    pub async fn initialize(&mut self) -> Result<GeminiAcpInitialize, GeminiAcpError> {
        let response = self
            .connection
            .request(
                "initialize",
                json!({
                    "protocolVersion": ACP_PROTOCOL_VERSION,
                    "clientInfo": {
                        "name": "elpis",
                        "title": "Elpis",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "clientCapabilities": {
                        "auth": { "terminal": false },
                        "fs": {
                            "readTextFile": false,
                            "writeTextFile": false
                        },
                        "terminal": false
                    }
                }),
                |_| {},
            )
            .await?;

        let protocol_version = response
            .get("protocolVersion")
            .and_then(Value::as_u64)
            .ok_or(GeminiAcpError::MissingField("protocolVersion"))?;
        if protocol_version != ACP_PROTOCOL_VERSION {
            return Err(GeminiAcpError::UnsupportedProtocol {
                expected: ACP_PROTOCOL_VERSION,
                actual: protocol_version,
            });
        }
        if response
            .pointer("/agentCapabilities/loadSession")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err(GeminiAcpError::MissingLoadSession);
        }

        let agent_name = response
            .pointer("/agentInfo/name")
            .and_then(Value::as_str)
            .unwrap_or("gemini-cli")
            .to_string();
        let agent_version = response
            .pointer("/agentInfo/version")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let auth_method_ids = response
            .get("authMethods")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|method| method.get("id").and_then(Value::as_str))
            .map(str::to_string)
            .collect();

        Ok(GeminiAcpInitialize {
            agent_name,
            agent_version,
            auth_method_ids,
        })
    }

    pub async fn authenticate(&mut self, method_id: &str) -> Result<(), GeminiAcpError> {
        self.connection
            .request("authenticate", json!({ "methodId": method_id }), |_| {})
            .await?;
        Ok(())
    }

    pub async fn new_session(&mut self, cwd: &Path) -> Result<GeminiAcpSession, GeminiAcpError> {
        let response = self
            .connection
            .request(
                "session/new",
                json!({
                    "cwd": cwd,
                    "mcpServers": []
                }),
                |_| {},
            )
            .await?;
        session_from_response(&response)
    }

    pub async fn load_session(
        &mut self,
        session_id: &str,
        cwd: &Path,
        mut on_notification: impl FnMut(Value),
    ) -> Result<(), GeminiAcpError> {
        self.connection
            .request(
                "session/load",
                json!({
                    "sessionId": session_id,
                    "cwd": cwd,
                    "mcpServers": []
                }),
                &mut on_notification,
            )
            .await?;
        Ok(())
    }

    pub async fn prompt(
        &mut self,
        session_id: &str,
        text: &str,
        mut on_notification: impl FnMut(Value),
    ) -> Result<Value, GeminiAcpError> {
        self.connection
            .request(
                "session/prompt",
                json!({
                    "sessionId": session_id,
                    "prompt": [{ "type": "text", "text": text }]
                }),
                &mut on_notification,
            )
            .await
    }

    pub async fn cancel(&mut self, session_id: &str) -> Result<(), GeminiAcpError> {
        self.connection
            .notify("session/cancel", json!({ "sessionId": session_id }))
            .await
    }

    pub async fn shutdown(mut self) -> Result<(), GeminiAcpError> {
        self.child.kill().await?;
        self.child.wait().await?;
        Ok(())
    }
}

fn session_from_response(response: &Value) -> Result<GeminiAcpSession, GeminiAcpError> {
    let session_id = response
        .get("sessionId")
        .and_then(Value::as_str)
        .ok_or(GeminiAcpError::MissingField("sessionId"))?;
    Ok(GeminiAcpSession {
        session_id: session_id.to_string(),
    })
}

struct AcpConnection<R, W> {
    lines: Lines<R>,
    writer: W,
    next_id: u64,
}

impl<R, W> AcpConnection<R, W>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    fn new(reader: R, writer: W) -> Self {
        Self {
            lines: reader.lines(),
            writer,
            next_id: 1,
        }
    }

    async fn request(
        &mut self,
        method: &str,
        params: Value,
        mut on_notification: impl FnMut(Value),
    ) -> Result<Value, GeminiAcpError> {
        let id = self.next_id;
        self.next_id += 1;
        self.write(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        }))
        .await?;

        loop {
            let line = self
                .lines
                .next_line()
                .await?
                .ok_or(GeminiAcpError::Closed)?;
            if line.trim().is_empty() {
                continue;
            }
            let message: Value = serde_json::from_str(&line)?;

            if let Some(response_id) = message.get("id")
                && message.get("method").is_none()
            {
                if response_id.as_u64() != Some(id) {
                    return Err(GeminiAcpError::UnexpectedResponseId {
                        expected: id,
                        actual: response_id.clone(),
                    });
                }
                if let Some(error) = message.get("error") {
                    return Err(GeminiAcpError::Rpc {
                        code: error.get("code").and_then(Value::as_i64).unwrap_or(-32000),
                        message: error
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown Gemini ACP error")
                            .to_string(),
                    });
                }
                return message
                    .get("result")
                    .cloned()
                    .ok_or(GeminiAcpError::MissingField("result"));
            }

            if let Some(request_id) = message.get("id")
                && message.get("method").is_some()
            {
                self.write(json!({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": "Elpis has not enabled this ACP client capability"
                    }
                }))
                .await?;
                continue;
            }

            if message.get("method").is_some() {
                on_notification(message);
            }
        }
    }

    async fn notify(&mut self, method: &str, params: Value) -> Result<(), GeminiAcpError> {
        self.write(json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }))
        .await
    }

    async fn write(&mut self, message: Value) -> Result<(), GeminiAcpError> {
        let mut encoded = serde_json::to_vec(&message)?;
        encoded.push(b'\n');
        self.writer.write_all(&encoded).await?;
        self.writer.flush().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;

    #[tokio::test]
    async fn request_collects_notifications_before_response() {
        let (client_io, agent_io) = duplex(4096);
        let (client_read, client_write) = tokio::io::split(client_io);
        let (agent_read, mut agent_write) = tokio::io::split(agent_io);
        let mut connection = AcpConnection::new(BufReader::new(client_read), client_write);

        let agent = tokio::spawn(async move {
            let mut request = String::new();
            BufReader::new(agent_read)
                .read_line(&mut request)
                .await
                .expect("read request");
            let request: Value = serde_json::from_str(&request).expect("valid request");
            assert_eq!(request["method"], "session/prompt");
            assert_eq!(request["params"]["prompt"][0]["text"], "hello");

            agent_write
                .write_all(
                    b"{\"jsonrpc\":\"2.0\",\"method\":\"session/update\",\"params\":{\"update\":{\"sessionUpdate\":\"agent_message_chunk\"}}}\n",
                )
                .await
                .expect("write notification");
            agent_write
                .write_all(
                    b"{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"stopReason\":\"end_turn\"}}\n",
                )
                .await
                .expect("write response");
        });

        let mut notifications = Vec::new();
        let result = connection
            .request(
                "session/prompt",
                json!({
                    "sessionId": "session-1",
                    "prompt": [{ "type": "text", "text": "hello" }]
                }),
                |notification| notifications.push(notification),
            )
            .await
            .expect("prompt response");

        agent.await.expect("agent task");
        assert_eq!(result["stopReason"], "end_turn");
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0]["method"], "session/update");
    }

    #[tokio::test]
    async fn request_rejects_unsupported_client_capability_without_hanging() {
        let (client_io, agent_io) = duplex(4096);
        let (client_read, client_write) = tokio::io::split(client_io);
        let (agent_read, mut agent_write) = tokio::io::split(agent_io);
        let mut connection = AcpConnection::new(BufReader::new(client_read), client_write);

        let agent = tokio::spawn(async move {
            let mut lines = BufReader::new(agent_read).lines();
            lines
                .next_line()
                .await
                .expect("read request")
                .expect("request");
            agent_write
                .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"fs/read_text_file\",\"params\":{}}\n")
                .await
                .expect("write client request");
            let rejection = lines
                .next_line()
                .await
                .expect("read rejection")
                .expect("rejection");
            let rejection: Value = serde_json::from_str(&rejection).expect("valid rejection");
            assert_eq!(rejection["id"], 9);
            assert_eq!(rejection["error"]["code"], -32601);
            agent_write
                .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n")
                .await
                .expect("write response");
        });

        connection
            .request("initialize", json!({}), |_| {})
            .await
            .expect("request completes");
        agent.await.expect("agent task");
    }

    #[tokio::test]
    async fn rpc_errors_are_visible() {
        let (client_io, agent_io) = duplex(1024);
        let (client_read, client_write) = tokio::io::split(client_io);
        let (agent_read, mut agent_write) = tokio::io::split(agent_io);
        let mut connection = AcpConnection::new(BufReader::new(client_read), client_write);

        tokio::spawn(async move {
            let mut request = String::new();
            BufReader::new(agent_read)
                .read_line(&mut request)
                .await
                .expect("read request");
            agent_write
                .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{\"code\":-32602,\"message\":\"bad session\"}}\n")
                .await
                .expect("write error");
        });

        let error = connection
            .request("session/load", json!({}), |_| {})
            .await
            .expect_err("RPC error should surface");
        assert!(matches!(
            error,
            GeminiAcpError::Rpc {
                code: -32602,
                message
            } if message == "bad session"
        ));
    }
}
