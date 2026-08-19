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

impl From<larql_vindex::ResolvedPatchContextDetector> for ContextDetectorSpec {
    fn from(value: larql_vindex::ResolvedPatchContextDetector) -> Self {
        let representation = match value.representation {
            larql_vindex::PatchContextRepresentation::MeanNoBos => ContextRepresentation::MeanNoBos,
        };
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
}
