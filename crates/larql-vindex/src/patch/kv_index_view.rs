//! Read-only `KvIndex` view combining quantised source bytes with patch rows.
//!
//! Q4K inference owns a fully loaded `VectorIndex` for attention/FFN bytes,
//! while an LQL session owns a separate `PatchedVindex` over the same artifact.
//! This adapter keeps the large mmaps shared and exposes only exact row-level
//! deltas to `larql-compute`.

use std::collections::BTreeSet;
use std::sync::Arc;

use larql_compute::{FfnRowOverride, KvIndex};

use crate::error::VindexError;
use crate::index::types::{FfnRowAccess, PatchOverrides};
use crate::{PatchedVindex, VectorIndex};

/// A fail-closed overlay view for one generation sequence.
///
/// Non-contextual patch rows are always visible. A row carrying a contextual
/// detector is visible only when its slot appears in `allowed_context_slots`,
/// which must come from the current prompt's frozen detector decision.
pub struct PatchedKvIndexView<'a> {
    source: &'a VectorIndex,
    patch: &'a PatchedVindex,
    contextual_slots: BTreeSet<(usize, usize)>,
    allowed_context_slots: BTreeSet<(usize, usize)>,
}

impl<'a> PatchedKvIndexView<'a> {
    pub fn new(
        source: &'a VectorIndex,
        patch: &'a PatchedVindex,
        allowed_context_slots: impl IntoIterator<Item = (usize, usize)>,
    ) -> Result<Self, VindexError> {
        if source.num_layers != patch.base.num_layers
            || source.hidden_size != patch.base.hidden_size
            || (0..source.num_layers)
                .any(|layer| source.num_features(layer) != patch.base.num_features(layer))
        {
            return Err(VindexError::Parse(
                "quantised source and patch overlay shapes do not match".into(),
            ));
        }
        Ok(Self {
            source,
            patch,
            contextual_slots: patch
                .context_detectors()
                .into_iter()
                .map(|detector| (detector.layer, detector.feature))
                .collect(),
            allowed_context_slots: allowed_context_slots.into_iter().collect(),
        })
    }

    fn slot_is_visible(&self, layer: usize, feature: usize) -> bool {
        let slot = (layer, feature);
        !self.contextual_slots.contains(&slot) || self.allowed_context_slots.contains(&slot)
    }
}

impl KvIndex for PatchedKvIndexView<'_> {
    fn num_features(&self, layer: usize) -> usize {
        self.source.num_features(layer)
    }

    fn attn_kquant_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 4]> {
        <VectorIndex as KvIndex>::attn_kquant_layer_data(self.source, layer)
    }

    fn attn_q8_layer_data(&self, layer: usize) -> Option<[(&[u8], &[f32]); 4]> {
        <VectorIndex as KvIndex>::attn_q8_layer_data(self.source, layer)
    }

    fn interleaved_kquant_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(&[u8], &str); larql_compute::FFN_COMPONENTS_PER_LAYER]> {
        <VectorIndex as KvIndex>::interleaved_kquant_layer_data(self.source, layer)
    }

    fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
        <VectorIndex as KvIndex>::interleaved_kquant_mmap_ref(self.source)
    }

    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
        <VectorIndex as KvIndex>::interleaved_q4_mmap_ref(self.source)
    }

    fn kquant_ffn_layer_once(&self, layer: usize, component: usize) -> Option<Arc<Vec<f32>>> {
        <VectorIndex as KvIndex>::kquant_ffn_layer_once(self.source, layer, component)
    }

    fn ffn_row_overrides(&self, layer: usize) -> Option<Vec<FfnRowOverride<'_>>> {
        let overrides: Vec<_> = self
            .patch
            .override_slots_at(layer)?
            .into_iter()
            .filter(|slot| self.slot_is_visible(layer, slot.feature))
            .map(|slot| FfnRowOverride {
                feature: slot.feature,
                tombstoned: slot.tombstoned,
                gate: self.patch.gate_override(layer, slot.feature),
                up: self.patch.up_override(layer, slot.feature),
                down: self.patch.down_override(layer, slot.feature),
            })
            .collect();
        (!overrides.is_empty()).then_some(overrides)
    }

    fn ffn_base_row_scaled_add(
        &self,
        layer: usize,
        component: usize,
        feature: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        <VectorIndex as FfnRowAccess>::ffn_row_scaled_add(
            self.source,
            layer,
            component,
            feature,
            alpha,
            out,
        )
    }

    fn vocab_size(&self) -> usize {
        self.source.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::FeatureMeta;
    use crate::patch::{PatchContextRepresentation, ResolvedPatchContextDetector};

    fn fixture() -> (VectorIndex, PatchedVindex) {
        let source = VectorIndex::new(vec![None], vec![None], 1, 2);
        let mut patch = PatchedVindex::new(source.clone());
        patch.insert_feature(
            0,
            1,
            vec![1.0, 0.0],
            FeatureMeta {
                top_token: "target".into(),
                top_token_id: 7,
                c_score: 1.0,
                top_k: Vec::new(),
            },
        );
        patch.set_up_vector(0, 1, vec![0.0, 1.0]);
        patch.set_down_vector(0, 1, vec![2.0, 3.0]);
        patch.context_detectors.insert(
            (0, 1),
            ResolvedPatchContextDetector {
                layer: 0,
                feature: 1,
                target_ids: vec![7, 8],
                detector_layer: 0,
                representation: PatchContextRepresentation::MeanNoBos,
                center: vec![0.0, 0.0],
                entity_direction: vec![1.0, 0.0],
                relation_direction: vec![0.0, 1.0],
                entity_threshold: 0.0,
                relation_threshold: 0.0,
                calibration: None,
                residual_adapter: None,
            },
        );
        (source, patch)
    }

    #[test]
    fn contextual_slot_is_fail_closed_then_visible_when_allowed() {
        let (source, patch) = fixture();
        let rejected = PatchedKvIndexView::new(&source, &patch, []).unwrap();
        assert!(rejected.ffn_row_overrides(0).is_none());

        let accepted = PatchedKvIndexView::new(&source, &patch, [(0, 1)]).unwrap();
        let rows = accepted.ffn_row_overrides(0).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].feature, 1);
        assert_eq!(rows[0].gate, Some(&[1.0, 0.0][..]));
        assert_eq!(rows[0].up, Some(&[0.0, 1.0][..]));
        assert_eq!(rows[0].down, Some(&[2.0, 3.0][..]));
    }

    #[test]
    fn mismatched_source_shape_is_rejected() {
        let (_, patch) = fixture();
        let wrong = VectorIndex::new(vec![None], vec![None], 1, 3);
        assert!(PatchedKvIndexView::new(&wrong, &patch, []).is_err());
    }
}
