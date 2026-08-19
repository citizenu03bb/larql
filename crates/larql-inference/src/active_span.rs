//! Session-local continuation spans armed by an executed patch slot.
//!
//! A compose INSERT currently installs only the first target token in an FFN
//! slot.  `ActiveSpan` keeps the remaining tokenizer IDs outside the model
//! weights and exposes a one-token-at-a-time logit mask.  It is armed only
//! when the naturally selected first token and an effective activation from
//! the corresponding installed `(layer, feature)` agree.

use crate::model::ModelWeights;
use larql_vindex::{GateIndex, FFN_GATE, FFN_UP};
use serde::{Deserialize, Serialize};

/// Stable session-local identity of an installed FFN slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PatchSlotId {
    pub layer: usize,
    pub feature: usize,
}

/// Exact effective activation of one live patched slot for one inference.
///
/// The input is the real pre-FFN residual captured by `WalkFfn`.  The formula
/// mirrors the override path: `phi(gate·x) * (up·x)` for gated FFNs.  For an
/// installed down override, `contribution_norm` is exactly
/// `abs(activation) * ||down||` before it is added to the residual stream.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PatchActivationEvent {
    pub slot: PatchSlotId,
    pub gate_score: f32,
    /// Cosine between the live pre-FFN residual and the gate override.
    /// `None` when the slot has no explicit gate vector to normalise.
    pub gate_cosine: Option<f32>,
    pub up_score: Option<f32>,
    pub activation: f32,
    pub contribution_norm: Option<f32>,
}

/// Complete continuation metadata resolved from an applied `.vlp` patch or a
/// freshly installed session edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveSpanSpec {
    pub slot: PatchSlotId,
    /// Complete target, including the first token already produced naturally.
    pub target_ids: Vec<u32>,
}

impl From<&larql_vindex::ResolvedPatchContextDetector> for ActiveSpanSpec {
    fn from(value: &larql_vindex::ResolvedPatchContextDetector) -> Self {
        Self {
            slot: PatchSlotId {
                layer: value.layer,
                feature: value.feature,
            },
            target_ids: value.target_ids.clone(),
        }
    }
}

/// Armed exact-token continuation.  `next_index` starts at 1 because token 0
/// was selected naturally and is part of the trigger evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveSpan {
    spec: ActiveSpanSpec,
    next_index: usize,
}

/// Stateful adapter for the `FnMut(&[u32], &mut Vec<f32>)` contract shared by
/// the constrained dense, Q4K, and KV-cached generation loops.
///
/// The first generated token remains unmasked. On the following callback it
/// arms from that naturally selected token plus the prefill's effective patch
/// activation evidence, then constrains each remaining target ID exactly.
#[derive(Debug, Clone)]
pub struct ActiveSpanConstraint {
    specs: Vec<ActiveSpanSpec>,
    events: Vec<PatchActivationEvent>,
    active: Option<ActiveSpan>,
    triggered_slot: Option<PatchSlotId>,
    observed_generated: usize,
}

impl ActiveSpanConstraint {
    pub fn new(specs: Vec<ActiveSpanSpec>, events: Vec<PatchActivationEvent>) -> Self {
        Self {
            specs,
            events,
            active: None,
            triggered_slot: None,
            observed_generated: 0,
        }
    }

    pub fn active_slot(&self) -> Option<PatchSlotId> {
        self.active.as_ref().map(ActiveSpan::slot)
    }

    pub fn triggered_slot(&self) -> Option<PatchSlotId> {
        self.triggered_slot
    }

    /// Consume newly generated IDs and mask the next pick when a span remains
    /// active. Safe if a generator repeats a callback without adding a token.
    pub fn apply(&mut self, generated_ids: &[u32], logits: &mut [f32]) {
        while self.observed_generated < generated_ids.len() {
            let token_id = generated_ids[self.observed_generated];
            if self.observed_generated == 0 {
                self.active = ActiveSpan::try_arm(token_id, &self.specs, &self.events);
                self.triggered_slot = self.active.as_ref().map(ActiveSpan::slot);
            } else if let Some(span) = &mut self.active {
                if !span.observe_emitted(token_id) {
                    self.active = None;
                }
            }
            self.observed_generated += 1;
        }
        if let Some(span) = &self.active {
            span.mask_logits(logits);
        }
    }
}

impl ActiveSpan {
    /// Select the strongest eligible installed slot and arm its continuation.
    ///
    /// The first token alone is insufficient evidence: the same `(layer,
    /// feature)` must have a positive effective contribution. E4 found that
    /// neither raw contribution nor gate cosine admits a scalar floor: the
    /// weakest positive paraphrase sits below same-entity and same-relation
    /// false positives. Strict positivity therefore remains only the E3
    /// mechanism test; production arming needs a separate contextual detector.
    pub fn try_arm(
        selected_id: u32,
        specs: &[ActiveSpanSpec],
        events: &[PatchActivationEvent],
    ) -> Option<Self> {
        specs
            .iter()
            .filter(|spec| {
                spec.target_ids.len() > 1 && spec.target_ids.first() == Some(&selected_id)
            })
            .filter_map(|spec| {
                let event = events.iter().find(|event| event.slot == spec.slot)?;
                let strength = event.contribution_norm.unwrap_or(0.0);
                (event.activation > 0.0 && strength > 0.0).then_some((spec, strength))
            })
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(spec, _)| Self {
                spec: spec.clone(),
                next_index: 1,
            })
    }

    pub fn slot(&self) -> PatchSlotId {
        self.spec.slot
    }

    pub fn expected_next(&self) -> Option<u32> {
        self.spec.target_ids.get(self.next_index).copied()
    }

    /// Hard-mask logits to the next exact target ID.  Returns `false` once
    /// the span is complete and the decoder is therefore unconstrained.
    pub fn mask_logits(&self, logits: &mut [f32]) -> bool {
        let Some(expected) = self.expected_next() else {
            return false;
        };
        for (id, logit) in logits.iter_mut().enumerate() {
            if id != expected as usize {
                *logit = f32::NEG_INFINITY;
            }
        }
        true
    }

    /// Advance after the decoder emitted a token.  A mismatch cancels the
    /// span; completing the final token releases it immediately.
    pub fn observe_emitted(&mut self, token_id: u32) -> bool {
        if self.expected_next() != Some(token_id) {
            self.next_index = self.spec.target_ids.len();
            return false;
        }
        self.next_index += 1;
        self.expected_next().is_some()
    }

    pub fn is_complete(&self) -> bool {
        self.expected_next().is_none()
    }
}

/// Recompute the effective activation for every live patched slot from the
/// actual pre-FFN residual captured during this inference.
pub fn patch_activation_events(
    weights: &ModelWeights,
    index: &dyn GateIndex,
    residuals: &[(usize, Vec<f32>)],
) -> Vec<PatchActivationEvent> {
    let gated = weights.arch.ffn_type() == larql_models::FfnType::Gated;
    let gelu = weights.arch.activation().uses_gelu_tanh_gate_up();
    let phi = |value: f32| {
        if gelu {
            crate::ffn::gelu_tanh(value)
        } else {
            value * crate::ffn::sigmoid(value)
        }
    };

    let mut events = Vec::new();
    for (layer, residual) in residuals {
        let Some(slots) = index.override_slots_at(*layer) else {
            continue;
        };
        for slot in slots.into_iter().filter(|slot| !slot.tombstoned) {
            let feature = slot.feature;
            let dot = |override_row: Option<&[f32]>, component| {
                override_row
                    .filter(|row| row.len() == residual.len())
                    .map(|row| row.iter().zip(residual).map(|(a, b)| a * b).sum())
                    .or_else(|| index.ffn_row_dot(*layer, component, feature, residual))
            };

            let gate_override = index.gate_override(*layer, feature);
            let gate_score = if gated {
                let Some(score) = dot(gate_override, FFN_GATE) else {
                    continue;
                };
                score
            } else {
                0.0
            };
            let gate_cosine = gate_override
                .filter(|gate| gate.len() == residual.len())
                .and_then(|gate| {
                    let gate_norm = gate.iter().map(|value| value * value).sum::<f32>().sqrt();
                    let residual_norm = residual
                        .iter()
                        .map(|value| value * value)
                        .sum::<f32>()
                        .sqrt();
                    let denom = gate_norm * residual_norm;
                    (denom > 0.0).then_some(gate_score / denom)
                });
            let Some(up_score) = dot(index.up_override(*layer, feature), FFN_UP) else {
                continue;
            };
            let activation = if gated {
                phi(gate_score) * up_score
            } else {
                phi(up_score)
            };
            let contribution_norm = index.down_override(*layer, feature).map(|down| {
                let down_norm = down.iter().map(|value| value * value).sum::<f32>().sqrt();
                activation.abs() * down_norm
            });
            events.push(PatchActivationEvent {
                slot: PatchSlotId {
                    layer: *layer,
                    feature,
                },
                gate_score,
                gate_cosine,
                up_score: gated.then_some(up_score),
                activation,
                contribution_norm,
            });
        }
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(layer: usize, feature: usize, ids: &[u32]) -> ActiveSpanSpec {
        ActiveSpanSpec {
            slot: PatchSlotId { layer, feature },
            target_ids: ids.to_vec(),
        }
    }

    fn event(layer: usize, feature: usize, activation: f32, strength: f32) -> PatchActivationEvent {
        PatchActivationEvent {
            slot: PatchSlotId { layer, feature },
            gate_score: 1.0,
            gate_cosine: Some(1.0),
            up_score: Some(1.0),
            activation,
            contribution_norm: Some(strength),
        }
    }

    #[test]
    fn arm_requires_token_and_slot_activation() {
        let specs = [spec(2, 7, &[10, 11, 12])];
        assert!(ActiveSpan::try_arm(99, &specs, &[event(2, 7, 1.0, 2.0)]).is_none());
        assert!(ActiveSpan::try_arm(10, &specs, &[event(2, 8, 1.0, 2.0)]).is_none());
        assert!(ActiveSpan::try_arm(10, &specs, &[event(2, 7, -1.0, 2.0)]).is_none());

        let armed = ActiveSpan::try_arm(10, &specs, &[event(2, 7, 1.0, 2.0)]).unwrap();
        assert_eq!(
            armed.slot(),
            PatchSlotId {
                layer: 2,
                feature: 7
            }
        );
        assert_eq!(armed.expected_next(), Some(11));
    }

    #[test]
    fn strongest_eligible_slot_wins_collision() {
        let specs = [spec(2, 7, &[10, 11]), spec(3, 8, &[10, 12])];
        let armed =
            ActiveSpan::try_arm(10, &specs, &[event(2, 7, 1.0, 2.0), event(3, 8, 1.0, 4.0)])
                .unwrap();
        assert_eq!(
            armed.slot(),
            PatchSlotId {
                layer: 3,
                feature: 8
            }
        );
        assert_eq!(armed.expected_next(), Some(12));
    }

    #[test]
    fn mask_advances_then_releases() {
        let mut span =
            ActiveSpan::try_arm(10, &[spec(2, 7, &[10, 11, 12])], &[event(2, 7, 1.0, 2.0)])
                .unwrap();
        let mut logits = vec![1.0; 16];
        assert!(span.mask_logits(&mut logits));
        assert!(logits[11].is_finite());
        assert!(logits
            .iter()
            .enumerate()
            .all(|(id, value)| id == 11 || value.is_infinite()));
        assert!(span.observe_emitted(11));
        assert_eq!(span.expected_next(), Some(12));
        assert!(!span.observe_emitted(12));
        assert!(span.is_complete());

        let original = vec![0.0, 1.0, 2.0];
        let mut released = original.clone();
        assert!(!span.mask_logits(&mut released));
        assert_eq!(released, original);
    }

    #[test]
    fn constrained_generation_adapter_arms_after_natural_first_token() {
        let mut constraint =
            ActiveSpanConstraint::new(vec![spec(2, 7, &[10, 11, 12])], vec![event(2, 7, 1.0, 2.0)]);

        let original = vec![1.0; 16];
        let mut first_logits = original.clone();
        constraint.apply(&[], &mut first_logits);
        assert_eq!(first_logits, original, "first pick must remain natural");

        let mut second_logits = vec![1.0; 16];
        constraint.apply(&[10], &mut second_logits);
        assert_eq!(
            constraint.active_slot(),
            Some(PatchSlotId {
                layer: 2,
                feature: 7
            })
        );
        assert!(second_logits[11].is_finite());
        assert!(second_logits
            .iter()
            .enumerate()
            .all(|(id, value)| id == 11 || value.is_infinite()));

        let mut third_logits = vec![1.0; 16];
        constraint.apply(&[10, 11], &mut third_logits);
        assert!(third_logits[12].is_finite());
        assert!(third_logits
            .iter()
            .enumerate()
            .all(|(id, value)| id == 12 || value.is_infinite()));

        let mut trailing_logits = original.clone();
        constraint.apply(&[10, 11, 12], &mut trailing_logits);
        assert_eq!(constraint.active_slot(), None);
        assert_eq!(trailing_logits, original, "completion must release decoder");
    }

    #[test]
    fn constrained_generation_adapter_stays_open_without_trigger_event() {
        let mut constraint = ActiveSpanConstraint::new(vec![spec(2, 7, &[10, 11])], Vec::new());
        let original = vec![1.0; 16];
        let mut logits = original.clone();
        constraint.apply(&[10], &mut logits);
        assert_eq!(constraint.active_slot(), None);
        assert_eq!(logits, original);
    }

    #[test]
    fn patch_event_uses_live_override_vectors_and_real_residual() {
        let weights = crate::test_utils::make_test_weights();
        let hidden = weights.hidden_size;
        let base = crate::test_utils::make_test_vindex(&weights);
        let mut patched = larql_vindex::PatchedVindex::new(base);
        let layer = 0;
        let feature = 3;
        patched.insert_feature(
            layer,
            feature,
            vec![0.25; hidden],
            larql_vindex::FeatureMeta {
                top_token: "target".into(),
                top_token_id: 10,
                c_score: 0.9,
                top_k: Vec::new(),
            },
        );
        patched.set_up_vector(layer, feature, vec![0.5; hidden]);
        patched.set_down_vector(layer, feature, vec![0.75; hidden]);
        let residual = vec![1.0; hidden];

        let events = patch_activation_events(&weights, &patched, &[(layer, residual)]);
        let event = events
            .iter()
            .find(|event| event.slot == PatchSlotId { layer, feature })
            .expect("installed slot event");

        assert_eq!(event.gate_score, hidden as f32 * 0.25);
        assert!((event.gate_cosine.unwrap() - 1.0).abs() < 1e-6);
        assert_eq!(event.up_score, Some(hidden as f32 * 0.5));
        assert!(event.activation.is_finite() && event.activation > 0.0);
        assert!(event.contribution_norm.unwrap() > event.activation);
    }
}
