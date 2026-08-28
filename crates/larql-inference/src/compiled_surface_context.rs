//! Exact product-bounded eligibility for contextual multi-token patches.
//!
//! Unlike [`crate::context_detector`], this module reads no model residuals.
//! A specification lists complete prompt token-ID suffixes ending at the
//! decision position. Matching is exact and every non-match fails closed.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::active_span::{ActiveSpanSpec, PatchSlotId};
use crate::context_detector::{TerminalResidualAdapterSpec, TerminalResidualApplication};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompiledSurfaceState {
    Fire,
    CompiledReject,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledSurfaceContextSpec {
    /// Canonical digest of the exact vindex checksum manifest.
    pub base_checksum: String,
    pub payload: PatchSlotId,
    /// Complete target, including the naturally selected first ID.
    pub target_ids: Vec<u32>,
    /// Complete supported prompt surfaces, tokenized through the bound model.
    /// Runtime matching is suffix-anchored at the decision position.
    pub prompt_suffixes: Vec<Vec<u32>>,
    pub terminal_residual_adapter: TerminalResidualAdapterSpec,
    #[serde(default)]
    pub provenance: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledSurfaceDecision {
    pub payload: PatchSlotId,
    pub state: CompiledSurfaceState,
    /// Index in `prompt_suffixes`; absent for `compiled_reject`.
    pub matched_suffix: Option<usize>,
}

impl CompiledSurfaceDecision {
    pub fn allowed(&self) -> bool {
        self.state == CompiledSurfaceState::Fire
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledSurfaceContextError(pub String);

impl std::fmt::Display for CompiledSurfaceContextError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CompiledSurfaceContextError {}

impl CompiledSurfaceContextSpec {
    pub fn active_span_spec(&self) -> ActiveSpanSpec {
        ActiveSpanSpec {
            slot: self.payload,
            target_ids: self.target_ids.clone(),
        }
    }

    /// Validate every product-bound coordinate before any forward mutates
    /// state. The caller supplies active artifact geometry so an external C0
    /// manifest cannot silently escape the `.vlp` checksum boundary.
    pub fn validate(
        &self,
        active_base_checksum: &str,
        hidden_size: usize,
        vocab_size: usize,
    ) -> Result<(), CompiledSurfaceContextError> {
        if self.base_checksum.is_empty() || self.base_checksum != active_base_checksum {
            return Err(CompiledSurfaceContextError(format!(
                "compiled surface context base_checksum mismatch: expected {}, active artifact is {active_base_checksum}",
                self.base_checksum
            )));
        }
        if self.target_ids.len() < 2 {
            return Err(CompiledSurfaceContextError(
                "compiled surface context requires at least two target IDs".into(),
            ));
        }
        if let Some(id) = self
            .target_ids
            .iter()
            .copied()
            .find(|id| *id as usize >= vocab_size)
        {
            return Err(CompiledSurfaceContextError(format!(
                "compiled surface target ID {id} is outside vocabulary size {vocab_size}"
            )));
        }
        if self.prompt_suffixes.is_empty() {
            return Err(CompiledSurfaceContextError(
                "compiled surface context requires at least one prompt suffix".into(),
            ));
        }
        let mut unique = BTreeSet::new();
        for (index, suffix) in self.prompt_suffixes.iter().enumerate() {
            if suffix.is_empty() {
                return Err(CompiledSurfaceContextError(format!(
                    "compiled prompt suffix {index} is empty"
                )));
            }
            if let Some(id) = suffix.iter().copied().find(|id| *id as usize >= vocab_size) {
                return Err(CompiledSurfaceContextError(format!(
                    "compiled prompt suffix {index} ID {id} is outside vocabulary size {vocab_size}"
                )));
            }
            if !unique.insert(suffix) {
                return Err(CompiledSurfaceContextError(format!(
                    "compiled prompt suffix {index} duplicates an earlier suffix"
                )));
            }
        }
        let adapter = &self.terminal_residual_adapter;
        if adapter.application != TerminalResidualApplication::PrefillLastPositionV1
            || adapter.delta.len() != hidden_size
            || adapter.delta.iter().any(|value| !value.is_finite())
            || !adapter.delta.iter().any(|value| *value != 0.0)
        {
            return Err(CompiledSurfaceContextError(
                "compiled surface context has an invalid terminal residual adapter".into(),
            ));
        }
        Ok(())
    }

    /// Select the longest exact suffix. Manifest order resolves equal-length
    /// ties, making the evidence stable even when suffix banks overlap.
    pub fn evaluate(&self, prompt_ids: &[u32]) -> CompiledSurfaceDecision {
        let matched_suffix = self
            .prompt_suffixes
            .iter()
            .enumerate()
            .filter(|(_, suffix)| prompt_ids.ends_with(suffix))
            .max_by(|(left_index, left), (right_index, right)| {
                left.len()
                    .cmp(&right.len())
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index);
        CompiledSurfaceDecision {
            payload: self.payload,
            state: if matched_suffix.is_some() {
                CompiledSurfaceState::Fire
            } else {
                CompiledSurfaceState::CompiledReject
            },
            matched_suffix,
        }
    }
}

/// Evaluate a complete C0 manifest after atomic validation.
pub fn evaluate_compiled_surface_contexts(
    specs: &[CompiledSurfaceContextSpec],
    prompt_ids: &[u32],
    active_base_checksum: &str,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<Vec<CompiledSurfaceDecision>, CompiledSurfaceContextError> {
    let mut payloads = BTreeSet::new();
    for spec in specs {
        spec.validate(active_base_checksum, hidden_size, vocab_size)?;
        if !payloads.insert(spec.payload) {
            return Err(CompiledSurfaceContextError(format!(
                "compiled surface manifest repeats payload L{}/F{}",
                spec.payload.layer, spec.payload.feature
            )));
        }
    }
    Ok(specs.iter().map(|spec| spec.evaluate(prompt_ids)).collect())
}

/// Combine only adapters selected by explicit `fire` decisions.
pub fn compiled_surface_prefill_delta(
    specs: &[CompiledSurfaceContextSpec],
    decisions: &[CompiledSurfaceDecision],
    hidden_size: usize,
) -> Result<Option<Vec<f32>>, CompiledSurfaceContextError> {
    if specs.len() != decisions.len()
        || specs
            .iter()
            .zip(decisions)
            .any(|(spec, decision)| spec.payload != decision.payload)
    {
        return Err(CompiledSurfaceContextError(
            "compiled surface decisions do not match their manifest".into(),
        ));
    }
    let mut delta = vec![0.0f32; hidden_size];
    let mut selected = false;
    for (spec, decision) in specs.iter().zip(decisions) {
        if !decision.allowed() {
            continue;
        }
        let adapter = &spec.terminal_residual_adapter;
        if adapter.delta.len() != hidden_size
            || adapter.delta.iter().any(|value| !value.is_finite())
        {
            return Err(CompiledSurfaceContextError(format!(
                "invalid compiled surface adapter for L{}/F{}",
                spec.payload.layer, spec.payload.feature
            )));
        }
        for (sum, value) in delta.iter_mut().zip(&adapter.delta) {
            *sum += *value;
        }
        selected = true;
    }
    if !selected {
        return Ok(None);
    }
    if delta.iter().any(|value| !value.is_finite()) {
        return Err(CompiledSurfaceContextError(
            "combined compiled surface adapter is non-finite".into(),
        ));
    }
    Ok(Some(delta))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> CompiledSurfaceContextSpec {
        CompiledSurfaceContextSpec {
            base_checksum: "artifact".into(),
            payload: PatchSlotId {
                layer: 3,
                feature: 7,
            },
            target_ids: vec![10, 11, 12],
            prompt_suffixes: vec![vec![2, 3], vec![1, 2, 3], vec![4, 3]],
            terminal_residual_adapter: TerminalResidualAdapterSpec {
                application: TerminalResidualApplication::PrefillLastPositionV1,
                delta: vec![0.5, -0.5],
                calibration: Some("product-only".into()),
            },
            provenance: Some("fixture".into()),
        }
    }

    #[test]
    fn exact_suffix_match_fires_and_non_match_rejects() {
        let candidate = spec();
        let fire = candidate.evaluate(&[99, 1, 2, 3]);
        assert_eq!(fire.state, CompiledSurfaceState::Fire);
        assert_eq!(fire.matched_suffix, Some(1), "longest suffix wins");

        let reject = candidate.evaluate(&[1, 2, 3, 99]);
        assert_eq!(reject.state, CompiledSurfaceState::CompiledReject);
        assert_eq!(reject.matched_suffix, None);
    }

    #[test]
    fn equal_length_tie_uses_manifest_order() {
        let mut candidate = spec();
        candidate.prompt_suffixes = vec![vec![2, 3], vec![2, 3]];
        assert_eq!(candidate.evaluate(&[1, 2, 3]).matched_suffix, Some(0));
    }

    #[test]
    fn validation_rejects_malformed_or_unbound_specs() {
        let mut candidate = spec();
        candidate.prompt_suffixes.push(vec![2, 3]);
        assert!(candidate.validate("artifact", 2, 20).is_err());

        let mut candidate = spec();
        candidate.terminal_residual_adapter.delta = vec![0.0, 0.0];
        assert!(candidate.validate("artifact", 2, 20).is_err());

        assert!(spec().validate("wrong", 2, 20).is_err());
    }

    #[test]
    fn rejected_specs_contribute_no_terminal_delta() {
        let candidate = spec();
        let decisions = evaluate_compiled_surface_contexts(
            std::slice::from_ref(&candidate),
            &[8, 9],
            "artifact",
            2,
            20,
        )
        .unwrap();
        assert_eq!(
            compiled_surface_prefill_delta(&[candidate], &decisions, 2).unwrap(),
            None
        );
    }

    #[test]
    fn fired_specs_contribute_their_terminal_delta() {
        let candidate = spec();
        let decisions = evaluate_compiled_surface_contexts(
            std::slice::from_ref(&candidate),
            &[9, 2, 3],
            "artifact",
            2,
            20,
        )
        .unwrap();
        assert_eq!(
            compiled_surface_prefill_delta(&[candidate], &decisions, 2).unwrap(),
            Some(vec![0.5, -0.5])
        );
    }
}
