use clap::ValueEnum;

/// The low-level agent runtime selected for the current TUI session.
///
/// Additional variants belong here only after their transport, authentication, and
/// capability ownership have an approved contract and a working adapter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum RuntimeKind {
    /// Codex owns the native model loop, tools, thread, and compaction.
    #[default]
    Codex,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapabilityOwner {
    Elpis,
    Runtime,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeOwnership {
    pub turn: CapabilityOwner,
    pub native_tools: CapabilityOwner,
    pub native_thread: CapabilityOwner,
    pub native_compaction: CapabilityOwner,
    pub context_projection: CapabilityOwner,
    pub durable_memory: CapabilityOwner,
    pub session_mirror: CapabilityOwner,
    pub behavioral_policy: CapabilityOwner,
    pub evidence: CapabilityOwner,
}

impl RuntimeKind {
    pub const fn ownership(self) -> RuntimeOwnership {
        match self {
            Self::Codex => RuntimeOwnership {
                turn: CapabilityOwner::Runtime,
                native_tools: CapabilityOwner::Runtime,
                native_thread: CapabilityOwner::Runtime,
                native_compaction: CapabilityOwner::Runtime,
                context_projection: CapabilityOwner::Elpis,
                durable_memory: CapabilityOwner::Elpis,
                session_mirror: CapabilityOwner::Elpis,
                behavioral_policy: CapabilityOwner::Elpis,
                evidence: CapabilityOwner::Elpis,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codex_ownership_preserves_the_elpis_control_boundary() {
        let ownership = RuntimeKind::Codex.ownership();

        assert_eq!(ownership.turn, CapabilityOwner::Runtime);
        assert_eq!(ownership.native_tools, CapabilityOwner::Runtime);
        assert_eq!(ownership.native_thread, CapabilityOwner::Runtime);
        assert_eq!(ownership.native_compaction, CapabilityOwner::Runtime);
        assert_eq!(ownership.context_projection, CapabilityOwner::Elpis);
        assert_eq!(ownership.durable_memory, CapabilityOwner::Elpis);
        assert_eq!(ownership.session_mirror, CapabilityOwner::Elpis);
        assert_eq!(ownership.behavioral_policy, CapabilityOwner::Elpis);
        assert_eq!(ownership.evidence, CapabilityOwner::Elpis);
    }
}
