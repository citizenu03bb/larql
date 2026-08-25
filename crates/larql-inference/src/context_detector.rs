//! Model-bound contextual eligibility for conditional patch payloads.
//!
//! A detector observes an early pre-FFN residual matrix and decides whether a
//! later patched slot may replace its base contribution. Parameters are data,
//! never architecture constants. Live decisions belong to one forward pass;
//! persistence is a `.vlp` concern handled above this module.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::active_span::PatchSlotId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextRepresentation {
    MeanNoBos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalResidualApplication {
    PrefillLastPositionV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerminalResidualAdapterSpec {
    pub application: TerminalResidualApplication,
    pub delta: Vec<f32>,
    #[serde(default)]
    pub calibration: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextDetectorSpec {
    pub detector_layer: usize,
    pub representation: ContextRepresentation,
    pub center: Vec<f32>,
    pub entity_direction: Vec<f32>,
    pub relation_direction: Vec<f32>,
    pub entity_threshold: f32,
    pub relation_threshold: f32,
    pub payload: PatchSlotId,
    #[serde(default)]
    pub terminal_residual_adapter: Option<TerminalResidualAdapterSpec>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ContextDecision {
    pub payload: PatchSlotId,
    pub detector_layer: usize,
    pub entity_margin: f32,
    pub relation_margin: f32,
    pub allowed: bool,
}

impl ContextDetectorSpec {
    /// Evaluate one complete prompt prefill. `None` means malformed dimensions
    /// or no non-BOS position; callers must fail closed for the payload.
    pub fn evaluate(&self, residuals: &Array2<f32>) -> Option<ContextDecision> {
        let hidden = residuals.shape().get(1).copied()?;
        if residuals.shape()[0] <= 1
            || self.center.len() != hidden
            || self.entity_direction.len() != hidden
            || self.relation_direction.len() != hidden
        {
            return None;
        }
        let mut pooled = vec![0.0f32; hidden];
        match self.representation {
            ContextRepresentation::MeanNoBos => {
                for row in residuals.outer_iter().skip(1) {
                    for (sum, value) in pooled.iter_mut().zip(row) {
                        *sum += *value;
                    }
                }
                let denom = (residuals.shape()[0] - 1) as f32;
                for value in &mut pooled {
                    *value /= denom;
                }
            }
        }
        let centered = pooled.iter().zip(&self.center).map(|(x, c)| x - c);
        let entity_margin = centered
            .clone()
            .zip(&self.entity_direction)
            .map(|(x, direction)| x * direction)
            .sum::<f32>()
            - self.entity_threshold;
        let relation_margin = centered
            .zip(&self.relation_direction)
            .map(|(x, direction)| x * direction)
            .sum::<f32>()
            - self.relation_threshold;
        Some(ContextDecision {
            payload: self.payload,
            detector_layer: self.detector_layer,
            entity_margin,
            relation_margin,
            allowed: entity_margin >= 0.0 && relation_margin >= 0.0,
        })
    }
}

/// Resolve the terminal prefill delta from explicit per-slot decisions.
///
/// Missing or rejecting decisions contribute nothing. Every selected adapter
/// is validated before any caller mutates a hidden state, preserving atomic
/// fail-closed behaviour for manually constructed runtime specs as well as
/// already-validated `.vlp` patches.
pub fn terminal_prefill_delta(
    specs: &[ContextDetectorSpec],
    decisions: &[ContextDecision],
    hidden_size: usize,
) -> Result<Option<Vec<f32>>, String> {
    let mut selected = Vec::new();
    for spec in specs {
        let Some(adapter) = &spec.terminal_residual_adapter else {
            continue;
        };
        let explicitly_allowed = decisions
            .iter()
            .find(|decision| decision.payload == spec.payload)
            .is_some_and(|decision| decision.allowed);
        if !explicitly_allowed {
            continue;
        }
        if adapter.application != TerminalResidualApplication::PrefillLastPositionV1
            || adapter.delta.len() != hidden_size
            || adapter.delta.iter().any(|value| !value.is_finite())
            || !adapter.delta.iter().any(|value| *value != 0.0)
        {
            return Err(format!(
                "invalid terminal residual adapter for L{}/F{}",
                spec.payload.layer, spec.payload.feature
            ));
        }
        selected.push(adapter);
    }
    if selected.is_empty() {
        return Ok(None);
    }
    let mut delta = vec![0.0f32; hidden_size];
    for adapter in selected {
        for (sum, value) in delta.iter_mut().zip(&adapter.delta) {
            *sum += *value;
        }
    }
    if delta.iter().any(|value| !value.is_finite()) {
        return Err("combined terminal residual adapter is non-finite".into());
    }
    Ok(Some(delta))
}

/// Add the resolved delta to the final sequence row only.
pub fn apply_terminal_prefill_delta(
    hidden: &mut Array2<f32>,
    specs: &[ContextDetectorSpec],
    decisions: &[ContextDecision],
) -> Result<bool, String> {
    let hidden_size = hidden.ncols();
    let Some(delta) = terminal_prefill_delta(specs, decisions, hidden_size)? else {
        return Ok(false);
    };
    let Some(mut last) = hidden.rows_mut().into_iter().next_back() else {
        return Err("terminal residual adapter requires a non-empty prefill".into());
    };
    for (value, addition) in last.iter_mut().zip(delta) {
        *value += addition;
    }
    Ok(true)
}

impl From<larql_vindex::ResolvedPatchContextDetector> for ContextDetectorSpec {
    fn from(value: larql_vindex::ResolvedPatchContextDetector) -> Self {
        let representation = match value.representation {
            larql_vindex::PatchContextRepresentation::MeanNoBos => ContextRepresentation::MeanNoBos,
        };
        let terminal_residual_adapter = value.residual_adapter.map(|adapter| {
            let application = match adapter.application {
                larql_vindex::PatchResidualAdapterApplication::PrefillLastPositionV1 => {
                    TerminalResidualApplication::PrefillLastPositionV1
                }
            };
            TerminalResidualAdapterSpec {
                application,
                delta: adapter.delta,
                calibration: adapter.calibration,
            }
        });
        Self {
            detector_layer: value.detector_layer,
            representation,
            center: value.center,
            entity_direction: value.entity_direction,
            relation_direction: value.relation_direction,
            entity_threshold: value.entity_threshold,
            relation_threshold: value.relation_threshold,
            payload: PatchSlotId {
                layer: value.layer,
                feature: value.feature,
            },
            terminal_residual_adapter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ContextDetectorSpec {
        ContextDetectorSpec {
            detector_layer: 1,
            representation: ContextRepresentation::MeanNoBos,
            center: vec![1.0, 1.0],
            entity_direction: vec![1.0, 0.0],
            relation_direction: vec![0.0, 1.0],
            entity_threshold: 0.5,
            relation_threshold: 1.0,
            payload: PatchSlotId {
                layer: 3,
                feature: 7,
            },
            terminal_residual_adapter: None,
        }
    }

    #[test]
    fn mean_no_bos_requires_both_factor_margins() {
        let residuals =
            Array2::from_shape_vec((3, 2), vec![100.0, 100.0, 2.0, 3.0, 3.0, 2.0]).unwrap();
        let decision = spec().evaluate(&residuals).unwrap();
        assert_eq!(decision.entity_margin, 1.0);
        assert_eq!(decision.relation_margin, 0.5);
        assert!(decision.allowed);
    }

    #[test]
    fn mean_no_bos_rejects_when_one_factor_fails() {
        let residuals =
            Array2::from_shape_vec((3, 2), vec![100.0, 100.0, 2.0, 1.0, 3.0, 1.0]).unwrap();
        let decision = spec().evaluate(&residuals).unwrap();
        assert!(decision.entity_margin >= 0.0);
        assert!(decision.relation_margin < 0.0);
        assert!(!decision.allowed);
    }

    #[test]
    fn malformed_or_decode_only_input_fails_closed() {
        assert!(spec().evaluate(&Array2::zeros((1, 2))).is_none());
        assert!(spec().evaluate(&Array2::zeros((2, 3))).is_none());
    }

    fn decision(allowed: bool) -> ContextDecision {
        ContextDecision {
            payload: spec().payload,
            detector_layer: 1,
            entity_margin: 1.0,
            relation_margin: 1.0,
            allowed,
        }
    }

    fn spec_with_adapter() -> ContextDetectorSpec {
        ContextDetectorSpec {
            terminal_residual_adapter: Some(TerminalResidualAdapterSpec {
                application: TerminalResidualApplication::PrefillLastPositionV1,
                delta: vec![0.25, -0.5],
                calibration: Some("test".into()),
            }),
            ..spec()
        }
    }

    #[test]
    fn terminal_adapter_allow_changes_only_last_prefill_row() {
        let mut hidden = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(apply_terminal_prefill_delta(
            &mut hidden,
            &[spec_with_adapter()],
            &[decision(true)]
        )
        .unwrap());
        assert_eq!(
            hidden,
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.25, 3.5]).unwrap()
        );
    }

    #[test]
    fn terminal_adapter_reject_or_missing_is_bit_identical() {
        let original = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        for decisions in [Vec::new(), vec![decision(false)]] {
            let mut hidden = original.clone();
            assert!(
                !apply_terminal_prefill_delta(&mut hidden, &[spec_with_adapter()], &decisions)
                    .unwrap()
            );
            assert_eq!(hidden, original);
        }
    }

    #[test]
    fn legacy_spec_has_exact_terminal_parity() {
        let original = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut hidden = original.clone();
        assert!(!apply_terminal_prefill_delta(&mut hidden, &[spec()], &[decision(true)]).unwrap());
        assert_eq!(hidden, original);
    }
}
