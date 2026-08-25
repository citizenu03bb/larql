//! Patch application — `apply_patch`, `remove_patch`,
//! `rebuild_overrides` for `PatchedVindex`.
//!
//! Walks `VindexPatch::operations` and resolves each one into the
//! overlay's override maps (or the L0 KNN store for arch-B ops).
//! Pulled out of `overlay.rs` so the file holding `PatchedVindex`'s
//! query/mutation API stays focused.

use crate::error::VindexError;
use crate::index::types::DEFAULT_C_SCORE;
use crate::index::FeatureMeta;

use super::format::{
    decode_gate_vector, PatchContextDetector, PatchOp, PatchResidualAdapter,
    ResolvedPatchContextDetector, ResolvedPatchResidualAdapter, VindexPatch,
};
use super::overlay::PatchedVindex;

/// Decode-check every embedded vector in `patch` before any overlay
/// state is mutated, so a corrupt vector rejects the patch wholesale.
/// Without this, `apply_patch` used to drop the vector while still
/// applying the op's metadata — a silently half-applied patch.
// The wire operation and active artifact geometry are intentionally passed
// separately here so validation can name the exact failing field and remain
// atomic before overlay mutation.
#[allow(clippy::too_many_arguments)]
fn decode_context_detector(
    op_idx: usize,
    layer: usize,
    feature: usize,
    target_ids: &[u32],
    detector: &PatchContextDetector,
    residual_adapter: Option<&PatchResidualAdapter>,
    hidden_size: usize,
    num_layers: usize,
    vocab_size: usize,
) -> Result<ResolvedPatchContextDetector, VindexError> {
    if layer >= num_layers || detector.detector_layer >= layer {
        return Err(VindexError::Parse(format!(
            "patch op {op_idx}: contextual detector L{} must precede valid payload L{layer}",
            detector.detector_layer
        )));
    }
    let decode = |field: &str, b64: &str| -> Result<Vec<f32>, VindexError> {
        let vector = decode_gate_vector(b64)
            .map_err(|e| VindexError::Parse(format!("patch op {op_idx}: corrupt {field}: {e}")))?;
        if vector.len() != hidden_size || vector.iter().any(|value| !value.is_finite()) {
            return Err(VindexError::Parse(format!(
                "patch op {op_idx}: {field} must contain {hidden_size} finite f32 values"
            )));
        }
        Ok(vector)
    };
    if !detector.entity_threshold.is_finite() || !detector.relation_threshold.is_finite() {
        return Err(VindexError::Parse(format!(
            "patch op {op_idx}: contextual thresholds must be finite"
        )));
    }
    if target_ids.len() < 2 {
        return Err(VindexError::Parse(format!(
            "patch op {op_idx}: contextual multi-token payload must contain at least two target_ids"
        )));
    }
    if vocab_size > 0 {
        if let Some(target_id) = target_ids
            .iter()
            .copied()
            .find(|target_id| *target_id as usize >= vocab_size)
        {
            return Err(VindexError::Parse(format!(
                "patch op {op_idx}: target ID {target_id} is outside vocabulary size {vocab_size}"
            )));
        }
    }
    let residual_adapter = residual_adapter
        .map(|adapter| {
            let delta = decode(
                "residual_adapter.delta_vector_b64",
                &adapter.delta_vector_b64,
            )?;
            let norm_sq = delta
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>();
            if norm_sq <= 0.0 {
                return Err(VindexError::Parse(format!(
                    "patch op {op_idx}: residual adapter delta must have positive norm"
                )));
            }
            Ok(ResolvedPatchResidualAdapter {
                application: adapter.application,
                delta,
                calibration: adapter.calibration.clone(),
            })
        })
        .transpose()?;

    Ok(ResolvedPatchContextDetector {
        layer,
        feature,
        target_ids: target_ids.to_vec(),
        detector_layer: detector.detector_layer,
        representation: detector.representation,
        center: decode("center_vector_b64", &detector.center_vector_b64)?,
        entity_direction: decode("entity_direction_b64", &detector.entity_direction_b64)?,
        relation_direction: decode("relation_direction_b64", &detector.relation_direction_b64)?,
        entity_threshold: detector.entity_threshold,
        relation_threshold: detector.relation_threshold,
        calibration: detector.calibration.clone(),
        residual_adapter,
    })
}

fn validate_patch_vectors(
    patch: &VindexPatch,
    hidden_size: usize,
    num_layers: usize,
    vocab_size: usize,
) -> Result<(), VindexError> {
    fn check(op_idx: usize, field: &str, b64: &str) -> Result<(), VindexError> {
        decode_gate_vector(b64)
            .map(|_| ())
            .map_err(|e| VindexError::Parse(format!("patch op {op_idx}: corrupt {field}: {e}")))
    }
    for (i, op) in patch.operations.iter().enumerate() {
        match op {
            PatchOp::Insert {
                gate_vector_b64,
                up_vector_b64,
                down_vector_b64,
                ..
            }
            | PatchOp::Update {
                gate_vector_b64,
                up_vector_b64,
                down_vector_b64,
                ..
            } => {
                for (field, b64) in [
                    ("gate_vector_b64", gate_vector_b64),
                    ("up_vector_b64", up_vector_b64),
                    ("down_vector_b64", down_vector_b64),
                ] {
                    if let Some(b64) = b64 {
                        check(i, field, b64)?;
                    }
                }
            }
            PatchOp::InsertKnn { key_vector_b64, .. } => {
                check(i, "key_vector_b64", key_vector_b64)?;
            }
            PatchOp::Context {
                layer,
                feature,
                target_ids,
                detector,
                residual_adapter,
            } => {
                decode_context_detector(
                    i,
                    *layer,
                    *feature,
                    target_ids,
                    detector,
                    residual_adapter.as_ref(),
                    hidden_size,
                    num_layers,
                    vocab_size,
                )?;
                if residual_adapter.is_some()
                    && !patch.operations.iter().any(|candidate| {
                        matches!(
                            candidate,
                            PatchOp::Insert {
                                layer: insert_layer,
                                feature: insert_feature,
                                ..
                            } if insert_layer == layer && insert_feature == feature
                        )
                    })
                {
                    return Err(VindexError::Parse(format!(
                        "patch op {i}: residual adapter requires an Insert at payload L{layer}/F{feature}"
                    )));
                }
                if let Some(first_id) = target_ids.first() {
                    if let Some(insert_first_id) =
                        patch
                            .operations
                            .iter()
                            .find_map(|candidate| match candidate {
                                PatchOp::Insert {
                                    layer: insert_layer,
                                    feature: insert_feature,
                                    down_meta: Some(meta),
                                    ..
                                } if insert_layer == layer && insert_feature == feature => {
                                    Some(meta.top_token_id)
                                }
                                _ => None,
                            })
                    {
                        if *first_id != insert_first_id {
                            return Err(VindexError::Parse(format!(
                                "patch op {i}: first target ID {first_id} does not match payload L{layer}/F{feature} first ID {insert_first_id}"
                            )));
                        }
                    }
                }
            }
            PatchOp::Delete { .. } | PatchOp::DeleteKnn { .. } => {}
        }
    }
    Ok(())
}

impl PatchedVindex {
    fn validate_context_binding(
        patch: &VindexPatch,
        active_base_checksum: Option<&str>,
    ) -> Result<(), VindexError> {
        if !patch
            .operations
            .iter()
            .any(|op| matches!(op, PatchOp::Context { .. }))
        {
            return Ok(());
        }
        let expected = patch.base_checksum.as_deref().ok_or_else(|| {
            VindexError::Parse("contextual patch is missing its base_checksum binding".into())
        })?;
        let active = active_base_checksum.ok_or_else(|| {
            VindexError::Parse(
                "contextual patch requires a bound application with the active base checksum"
                    .into(),
            )
        })?;
        if expected != active {
            return Err(VindexError::Parse(format!(
                "contextual patch base_checksum mismatch: expected {expected}, active artifact is {active}"
            )));
        }
        Ok(())
    }

    /// Apply a patch, rejecting it wholesale if any embedded vector
    /// fails to decode. All-or-nothing: on `Err` no overlay state has
    /// been touched and the patch is not recorded.
    pub fn try_apply_patch(&mut self, patch: VindexPatch) -> Result<(), VindexError> {
        Self::validate_context_binding(&patch, None)?;
        self.try_apply_patch_validated_binding(patch)
    }

    /// Apply a patch while binding model-specific contextual coordinates to
    /// the active vindex artifact. Context-free legacy patches remain valid
    /// through [`Self::try_apply_patch`].
    pub fn try_apply_patch_bound(
        &mut self,
        patch: VindexPatch,
        active_base_checksum: Option<&str>,
    ) -> Result<(), VindexError> {
        Self::validate_context_binding(&patch, active_base_checksum)?;
        self.try_apply_patch_validated_binding(patch)
    }

    fn try_apply_patch_validated_binding(&mut self, patch: VindexPatch) -> Result<(), VindexError> {
        validate_patch_vectors(
            &patch,
            self.base.hidden_size,
            self.base.num_layers,
            self.base.vocab_size,
        )?;
        self.apply_patch_unchecked(patch);
        Ok(())
    }

    /// Apply a patch. Operations are resolved into the override maps.
    ///
    /// Infallible compatibility wrapper around [`try_apply_patch`]:
    /// a patch with a corrupt embedded vector is dropped **whole** (no
    /// state change, not recorded in `patches`) rather than
    /// half-applied. Callers that need to surface the decode error use
    /// `try_apply_patch`.
    ///
    /// [`try_apply_patch`]: Self::try_apply_patch
    pub fn apply_patch(&mut self, patch: VindexPatch) {
        let _ = self.try_apply_patch(patch);
    }

    /// Application body. Every `*_b64` field has already been decode-
    /// checked by `validate_patch_vectors`, so the inner `if let Ok`
    /// arms cannot silently skip a vector. Gate-snapshot invalidation
    /// is handled per-layer inside `GateOverlay`'s mutators.
    fn apply_patch_unchecked(&mut self, patch: VindexPatch) {
        for op in &patch.operations {
            match op {
                PatchOp::InsertKnn {
                    layer,
                    entity,
                    relation,
                    target,
                    target_id,
                    confidence,
                    key_vector_b64,
                } => {
                    if let Ok(key_vec) = decode_gate_vector(key_vector_b64) {
                        self.knn_store.add(
                            *layer,
                            key_vec,
                            *target_id,
                            target.clone(),
                            entity.clone(),
                            relation.clone(),
                            confidence.unwrap_or(1.0),
                        );
                    }
                    continue;
                }
                PatchOp::DeleteKnn { entity } => {
                    self.knn_store.remove_by_entity(entity);
                    continue;
                }
                PatchOp::Context {
                    layer,
                    feature,
                    target_ids,
                    detector,
                    residual_adapter,
                } => {
                    let resolved = decode_context_detector(
                        0,
                        *layer,
                        *feature,
                        target_ids,
                        detector,
                        residual_adapter.as_ref(),
                        self.base.hidden_size,
                        self.base.num_layers,
                        self.base.vocab_size,
                    )
                    .expect("context detector was validated before apply");
                    self.context_detectors.insert((*layer, *feature), resolved);
                    continue;
                }
                _ => {}
            }
            let key = op.key().unwrap(); // safe: only Arch A ops reach here
            match op {
                PatchOp::Insert {
                    target,
                    confidence,
                    gate_vector_b64,
                    up_vector_b64,
                    down_vector_b64,
                    down_meta,
                    ..
                } => {
                    let meta = if let Some(dm) = down_meta {
                        FeatureMeta {
                            top_token: dm.top_token.clone(),
                            top_token_id: dm.top_token_id,
                            c_score: dm.c_score,
                            top_k: vec![larql_models::TopKEntry {
                                token: dm.top_token.clone(),
                                token_id: dm.top_token_id,
                                logit: dm.c_score,
                            }],
                        }
                    } else {
                        FeatureMeta {
                            top_token: target.clone(),
                            top_token_id: 0,
                            c_score: confidence.unwrap_or(DEFAULT_C_SCORE),
                            top_k: vec![],
                        }
                    };
                    self.overrides_meta.insert(key, Some(meta));
                    self.deleted.remove(&key);
                    if let Some(b64) = gate_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_gate.insert(key.0, key.1, vec);
                        }
                    }
                    if let Some(b64) = up_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.base.set_up_vector(key.0, key.1, vec);
                        }
                    }
                    if let Some(b64) = down_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.base.set_down_vector(key.0, key.1, vec);
                        }
                    }
                }
                PatchOp::Update {
                    gate_vector_b64,
                    up_vector_b64,
                    down_vector_b64,
                    down_meta,
                    ..
                } => {
                    if let Some(dm) = down_meta {
                        let meta = FeatureMeta {
                            top_token: dm.top_token.clone(),
                            top_token_id: dm.top_token_id,
                            c_score: dm.c_score,
                            top_k: vec![larql_models::TopKEntry {
                                token: dm.top_token.clone(),
                                token_id: dm.top_token_id,
                                logit: dm.c_score,
                            }],
                        };
                        self.overrides_meta.insert(key, Some(meta));
                    }
                    // Resurrect: Update on a tombstoned slot implies
                    // the feature exists again (mirrors Insert; the
                    // `deleted` contract on `PatchedVindex`, review
                    // 2026-07-30 M6). If the earlier Delete pinned a
                    // `None` meta override and this Update carries no
                    // replacement meta, drop the pin so `feature_meta`
                    // falls through to the base — keeping
                    // `feature_meta()` and `gate_knn()` in agreement.
                    if self.deleted.remove(&key)
                        && matches!(self.overrides_meta.get(&key), Some(None))
                    {
                        self.overrides_meta.remove(&key);
                    }
                    if let Some(b64) = gate_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_gate.insert(key.0, key.1, vec);
                        }
                    }
                    if let Some(b64) = up_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.base.set_up_vector(key.0, key.1, vec);
                        }
                    }
                    if let Some(b64) = down_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.base.set_down_vector(key.0, key.1, vec);
                        }
                    }
                }
                PatchOp::Delete { .. } => {
                    self.overrides_meta.insert(key, None);
                    self.deleted.insert(key);
                    self.context_detectors.remove(&key);
                    // `GateOverlay::remove` invalidates the layer's
                    // snapshot even when the gate entry was absent —
                    // the per-layer feature set shrunk either way.
                    self.overrides_gate.remove(key.0, key.1);
                }
                PatchOp::InsertKnn { .. } | PatchOp::DeleteKnn { .. } | PatchOp::Context { .. } => {
                    unreachable!("KNN ops handled above");
                }
            }
        }
        self.patches.push(patch);
    }

    /// Remove the last applied patch and rebuild overrides.
    pub fn remove_patch(&mut self, index: usize) {
        if index < self.patches.len() {
            self.patches.remove(index);
            self.rebuild_overrides();
        }
    }

    /// Rebuild override maps from scratch (after removing a patch).
    fn rebuild_overrides(&mut self) {
        self.overrides_meta.clear();
        // Clears rows AND every layer's cached snapshot.
        self.overrides_gate.clear();
        self.deleted.clear();
        self.knn_store = super::knn_store::KnnStore::default();
        self.context_detectors.clear();
        let patches: Vec<VindexPatch> = self.patches.drain(..).collect();
        for patch in patches {
            // Recorded patches already passed binding validation. Rebuilding
            // only rechecks their payload integrity before replaying them.
            if validate_patch_vectors(
                &patch,
                self.base.hidden_size,
                self.base.num_layers,
                self.base.vocab_size,
            )
            .is_ok()
            {
                self.apply_patch_unchecked(patch);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::VectorIndex;
    use crate::patch::format::{
        encode_gate_vector, PatchContextDetector, PatchContextRepresentation, PatchDownMeta,
        PatchOp, PatchResidualAdapter, PatchResidualAdapterApplication, VindexPatch,
    };

    fn empty_pv() -> PatchedVindex {
        PatchedVindex::new(VectorIndex::new(vec![], vec![], 0, 0))
    }

    fn make_patch(ops: Vec<PatchOp>) -> VindexPatch {
        VindexPatch {
            version: 1,
            base_model: "test".into(),
            base_checksum: None,
            created_at: "2026-01-01T00:00:00Z".into(),
            description: None,
            author: None,
            tags: vec![],
            operations: ops,
        }
    }

    fn context_op(vector_len: usize) -> PatchOp {
        PatchOp::Context {
            layer: 3,
            feature: 7,
            target_ids: vec![11, 12, 13],
            detector: PatchContextDetector {
                detector_layer: 1,
                representation: PatchContextRepresentation::MeanNoBos,
                center_vector_b64: encode_gate_vector(&vec![0.0; vector_len]),
                entity_direction_b64: encode_gate_vector(&vec![1.0; vector_len]),
                relation_direction_b64: encode_gate_vector(&vec![2.0; vector_len]),
                entity_threshold: -0.25,
                relation_threshold: 0.75,
                calibration: Some("test".into()),
            },
            residual_adapter: None,
        }
    }

    fn context_insert() -> PatchOp {
        PatchOp::Insert {
            layer: 3,
            feature: 7,
            relation: Some("ruler".into()),
            entity: "Lothlorien".into(),
            target: "Galadriel".into(),
            confidence: Some(1.0),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(PatchDownMeta {
                top_token: "Gal".into(),
                top_token_id: 11,
                c_score: 1.0,
            }),
        }
    }

    fn context_op_with_adapter(delta: &[f32]) -> PatchOp {
        let mut op = context_op(2);
        let PatchOp::Context {
            residual_adapter, ..
        } = &mut op
        else {
            unreachable!()
        };
        *residual_adapter = Some(PatchResidualAdapter {
            application: PatchResidualAdapterApplication::PrefillLastPositionV1,
            delta_vector_b64: encode_gate_vector(delta),
            calibration: Some("spent-dev-only".into()),
        });
        op
    }

    fn contextual_pv() -> PatchedVindex {
        PatchedVindex::new(VectorIndex::new(vec![None; 4], vec![None; 4], 4, 2))
    }

    fn contextual_patch(ops: Vec<PatchOp>) -> VindexPatch {
        let mut patch = make_patch(ops);
        patch.base_checksum = Some("sha256-manifest:test".into());
        patch
    }

    #[test]
    fn apply_context_decodes_validates_and_exposes_detector() {
        let mut pv = contextual_pv();
        pv.try_apply_patch_bound(
            contextual_patch(vec![context_op(2)]),
            Some("sha256-manifest:test"),
        )
        .unwrap();
        let detectors = pv.context_detectors();
        assert_eq!(detectors.len(), 1);
        assert_eq!((detectors[0].layer, detectors[0].feature), (3, 7));
        assert_eq!(detectors[0].detector_layer, 1);
        assert_eq!(detectors[0].target_ids, vec![11, 12, 13]);
        assert_eq!(detectors[0].entity_direction, vec![1.0, 1.0]);
        assert!(detectors[0].residual_adapter.is_none());
    }

    #[test]
    fn contextual_residual_adapter_resolves_on_linked_insert() {
        let mut pv = contextual_pv();
        pv.try_apply_patch_bound(
            contextual_patch(vec![
                context_insert(),
                context_op_with_adapter(&[0.25, -0.5]),
            ]),
            Some("sha256-manifest:test"),
        )
        .unwrap();
        let detectors = pv.context_detectors();
        let adapter = detectors[0].residual_adapter.as_ref().unwrap();
        assert_eq!(
            adapter.application,
            PatchResidualAdapterApplication::PrefillLastPositionV1
        );
        assert_eq!(adapter.delta, vec![0.25, -0.5]);
        assert_eq!(adapter.calibration.as_deref(), Some("spent-dev-only"));
    }

    #[test]
    fn contextual_residual_adapter_requires_insert_in_same_patch() {
        let mut pv = contextual_pv();
        let error = pv
            .try_apply_patch_bound(
                contextual_patch(vec![context_op_with_adapter(&[0.25, -0.5])]),
                Some("sha256-manifest:test"),
            )
            .unwrap_err();
        assert!(error.to_string().contains("requires an Insert"));
        assert!(pv.patches.is_empty());
        assert!(pv.context_detectors().is_empty());

        let mut pv = contextual_pv();
        let mut wrong_slot = context_insert();
        let PatchOp::Insert { feature, .. } = &mut wrong_slot else {
            unreachable!()
        };
        *feature = 6;
        let error = pv
            .try_apply_patch_bound(
                contextual_patch(vec![wrong_slot, context_op_with_adapter(&[0.25, -0.5])]),
                Some("sha256-manifest:test"),
            )
            .unwrap_err();
        assert!(error.to_string().contains("requires an Insert"));
        assert!(pv.patches.is_empty());
        assert!(pv.overrides_meta.is_empty());
    }

    #[test]
    fn invalid_residual_adapter_vectors_reject_patch_atomically() {
        for (delta, expected) in [
            (vec![0.0, 0.0], "positive norm"),
            (vec![1.0, 2.0, 3.0], "must contain 2 finite f32 values"),
            (vec![f32::NAN, 1.0], "must contain 2 finite f32 values"),
        ] {
            let mut pv = contextual_pv();
            let error = pv
                .try_apply_patch_bound(
                    contextual_patch(vec![context_insert(), context_op_with_adapter(&delta)]),
                    Some("sha256-manifest:test"),
                )
                .unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
            assert!(pv.patches.is_empty());
            assert!(pv.overrides_meta.is_empty());
            assert!(pv.context_detectors().is_empty());
        }
    }

    #[test]
    fn removing_adapter_patch_clears_resolved_context_state() {
        let mut pv = contextual_pv();
        pv.try_apply_patch_bound(
            contextual_patch(vec![
                context_insert(),
                context_op_with_adapter(&[0.25, -0.5]),
            ]),
            Some("sha256-manifest:test"),
        )
        .unwrap();
        assert!(pv.context_detectors()[0].residual_adapter.is_some());
        pv.remove_patch(0);
        assert!(pv.context_detectors().is_empty());
    }

    #[test]
    fn invalid_context_dimension_rejects_patch_wholesale() {
        let mut pv = contextual_pv();
        let patch = contextual_patch(vec![
            PatchOp::Insert {
                layer: 0,
                feature: 1,
                relation: None,
                entity: "A".into(),
                target: "B".into(),
                confidence: None,
                gate_vector_b64: None,
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            },
            context_op(3),
        ]);
        assert!(pv
            .try_apply_patch_bound(patch, Some("sha256-manifest:test"))
            .is_err());
        assert!(pv.patches.is_empty());
        assert!(pv.overrides_meta.is_empty());
        assert!(pv.context_detectors().is_empty());
    }

    #[test]
    fn contextual_patch_requires_matching_bound_artifact() {
        let mut pv = contextual_pv();
        let patch = contextual_patch(vec![context_op(2)]);
        assert!(pv.try_apply_patch(patch.clone()).is_err());
        assert!(pv
            .try_apply_patch_bound(patch, Some("sha256-manifest:wrong"))
            .is_err());
        assert!(pv.patches.is_empty());
    }

    #[test]
    fn invalid_context_target_ids_reject_patch_wholesale() {
        let mut pv = contextual_pv();
        let mut op = context_op(2);
        let PatchOp::Context { target_ids, .. } = &mut op else {
            unreachable!()
        };
        target_ids.truncate(1);
        let error = pv
            .try_apply_patch_bound(contextual_patch(vec![op]), Some("sha256-manifest:test"))
            .unwrap_err();
        assert!(error.to_string().contains("at least two target_ids"));
        assert!(pv.patches.is_empty());
        assert!(pv.context_detectors().is_empty());
    }

    #[test]
    fn context_first_target_must_match_insert_metadata() {
        let mut pv = contextual_pv();
        let insert = PatchOp::Insert {
            layer: 3,
            feature: 7,
            relation: None,
            entity: "Lothlorien".into(),
            target: "Galadriel".into(),
            confidence: None,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(PatchDownMeta {
                top_token: "Gal".into(),
                top_token_id: 99,
                c_score: 1.0,
            }),
        };
        let error = pv
            .try_apply_patch_bound(
                contextual_patch(vec![insert, context_op(2)]),
                Some("sha256-manifest:test"),
            )
            .unwrap_err();
        assert!(error.to_string().contains("does not match payload"));
        assert!(pv.patches.is_empty());
        assert!(pv.overrides_meta.is_empty());
        assert!(pv.context_detectors().is_empty());
    }

    #[test]
    fn context_target_ids_must_fit_known_vocabulary() {
        let mut pv = contextual_pv();
        pv.base_mut().vocab_size = 12;
        let error = pv
            .try_apply_patch_bound(
                contextual_patch(vec![context_op(2)]),
                Some("sha256-manifest:test"),
            )
            .unwrap_err();
        assert!(error.to_string().contains("outside vocabulary size 12"));
        assert!(pv.patches.is_empty());
    }

    #[test]
    fn apply_insert_populates_overrides_meta() {
        let mut pv = empty_pv();
        let patch = make_patch(vec![PatchOp::Insert {
            layer: 2,
            feature: 5,
            relation: None,
            entity: "France".into(),
            target: "Paris".into(),
            confidence: Some(0.9),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]);
        pv.apply_patch(patch);
        assert!(pv.overrides_meta.contains_key(&(2, 5)));
        let meta = pv.overrides_meta[&(2, 5)].as_ref().unwrap();
        assert_eq!(meta.top_token, "Paris");
    }

    #[test]
    fn apply_insert_with_down_meta_uses_down_meta_token() {
        let mut pv = empty_pv();
        let patch = make_patch(vec![PatchOp::Insert {
            layer: 1,
            feature: 10,
            relation: None,
            entity: "Germany".into(),
            target: "Berlin".into(),
            confidence: Some(0.8),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(PatchDownMeta {
                top_token: "Berlin".into(),
                top_token_id: 42,
                c_score: 0.75,
            }),
        }]);
        pv.apply_patch(patch);
        let meta = pv.overrides_meta[&(1, 10)].as_ref().unwrap();
        assert_eq!(meta.top_token, "Berlin");
        assert_eq!(meta.top_token_id, 42);
        assert!((meta.c_score - 0.75).abs() < 1e-6);
    }

    #[test]
    fn apply_insert_with_gate_vector_populates_overrides_gate() {
        let mut pv = empty_pv();
        let gv = vec![1.0f32, 0.0, -1.0];
        let b64 = encode_gate_vector(&gv);
        let patch = make_patch(vec![PatchOp::Insert {
            layer: 3,
            feature: 7,
            relation: None,
            entity: "Spain".into(),
            target: "Madrid".into(),
            confidence: None,
            gate_vector_b64: Some(b64),
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]);
        pv.apply_patch(patch);
        assert!(pv.overrides_gate.contains(3, 7));
        let stored = pv.overrides_gate.get(3, 7).unwrap();
        assert_eq!(stored.len(), 3);
        assert_eq!(stored[0].to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn apply_insert_with_up_and_down_vectors_populates_base_overrides() {
        // Compose-mode INSERT writes gate + up + down overrides; the .vlp
        // must round-trip all three. Without up_vector_b64 /
        // down_vector_b64 in the patch, re-applying the file (e.g. on
        // `larql apply` after a save) would lose up + down and
        // `COMPILE INTO VINDEX` would bake nothing.
        let mut pv = empty_pv();
        let gate = vec![1.0f32, 2.0, 3.0];
        let up = vec![0.1f32, 0.2, 0.3];
        let down = vec![-0.5f32, 0.0, 0.5];
        let patch = make_patch(vec![PatchOp::Insert {
            layer: 4,
            feature: 9,
            relation: Some("capital".into()),
            entity: "France".into(),
            target: "Paris".into(),
            confidence: Some(0.9),
            gate_vector_b64: Some(encode_gate_vector(&gate)),
            up_vector_b64: Some(encode_gate_vector(&up)),
            down_vector_b64: Some(encode_gate_vector(&down)),
            down_meta: None,
        }]);
        pv.apply_patch(patch);
        assert_eq!(pv.overrides_gate_at(4, 9), Some(gate.as_slice()));
        assert_eq!(pv.up_override_at(4, 9), Some(up.as_slice()));
        assert_eq!(pv.down_override_at(4, 9), Some(down.as_slice()));
    }

    #[test]
    fn apply_delete_tombstones_feature() {
        let mut pv = empty_pv();
        let patch = make_patch(vec![PatchOp::Delete {
            layer: 0,
            feature: 3,
            reason: None,
        }]);
        pv.apply_patch(patch);
        assert!(pv.deleted.contains(&(0, 3)));
        assert!(pv.overrides_meta[&(0, 3)].is_none());
    }

    #[test]
    fn insert_then_delete_removes_gate_override() {
        let mut pv = empty_pv();
        let gv = vec![1.0f32, 2.0];
        let b64 = encode_gate_vector(&gv);
        let insert_patch = make_patch(vec![PatchOp::Insert {
            layer: 0,
            feature: 1,
            relation: None,
            entity: "A".into(),
            target: "B".into(),
            confidence: None,
            gate_vector_b64: Some(b64),
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]);
        pv.apply_patch(insert_patch);
        assert!(pv.overrides_gate.contains(0, 1));

        let delete_patch = make_patch(vec![PatchOp::Delete {
            layer: 0,
            feature: 1,
            reason: None,
        }]);
        pv.apply_patch(delete_patch);
        assert!(!pv.overrides_gate.contains(0, 1));
        assert!(pv.deleted.contains(&(0, 1)));
    }

    #[test]
    fn apply_update_sets_meta_only() {
        let mut pv = empty_pv();
        let patch = make_patch(vec![PatchOp::Update {
            layer: 0,
            feature: 2,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(PatchDownMeta {
                top_token: "updated".into(),
                top_token_id: 99,
                c_score: 0.5,
            }),
        }]);
        pv.apply_patch(patch);
        let meta = pv.overrides_meta[&(0, 2)].as_ref().unwrap();
        assert_eq!(meta.top_token, "updated");
        // No gate override set
        assert!(!pv.overrides_gate.contains(0, 2));
    }

    #[test]
    fn apply_patches_accumulate_in_order() {
        let mut pv = empty_pv();
        let p1 = make_patch(vec![PatchOp::Insert {
            layer: 0,
            feature: 0,
            relation: None,
            entity: "X".into(),
            target: "Y".into(),
            confidence: Some(0.5),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]);
        let p2 = make_patch(vec![PatchOp::Insert {
            layer: 0,
            feature: 1,
            relation: None,
            entity: "A".into(),
            target: "B".into(),
            confidence: Some(0.9),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]);
        pv.apply_patch(p1);
        pv.apply_patch(p2);
        assert_eq!(pv.patches.len(), 2);
        assert!(pv.overrides_meta.contains_key(&(0, 0)));
        assert!(pv.overrides_meta.contains_key(&(0, 1)));
    }

    #[test]
    fn remove_patch_rebuilds_overrides() {
        let mut pv = empty_pv();
        let p1 = make_patch(vec![PatchOp::Insert {
            layer: 0,
            feature: 5,
            relation: None,
            entity: "X".into(),
            target: "first".into(),
            confidence: None,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]);
        let p2 = make_patch(vec![PatchOp::Insert {
            layer: 0,
            feature: 6,
            relation: None,
            entity: "Y".into(),
            target: "second".into(),
            confidence: None,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]);
        pv.apply_patch(p1);
        pv.apply_patch(p2);
        assert_eq!(pv.patches.len(), 2);

        pv.remove_patch(0);
        assert_eq!(pv.patches.len(), 1);
        // Feature 5 (from patch 0) should be gone
        assert!(!pv.overrides_meta.contains_key(&(0, 5)));
        // Feature 6 (from patch 1) should still be present
        assert!(pv.overrides_meta.contains_key(&(0, 6)));
    }

    #[test]
    fn remove_patch_out_of_bounds_is_noop() {
        let mut pv = empty_pv();
        pv.remove_patch(999); // should not panic
        assert!(pv.patches.is_empty());
    }

    #[test]
    fn try_apply_patch_rejects_corrupt_vector_without_half_applying() {
        // Regression: a corrupt gate vector used to be silently dropped
        // while the op's metadata still landed — a half-applied patch.
        // The whole patch must be rejected and no state mutated,
        // including ops *before* the corrupt one.
        let mut pv = empty_pv();
        let patch = make_patch(vec![
            PatchOp::Insert {
                layer: 0,
                feature: 0,
                relation: None,
                entity: "A".into(),
                target: "B".into(),
                confidence: Some(0.9),
                gate_vector_b64: Some(encode_gate_vector(&[1.0, 2.0])),
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            },
            PatchOp::Insert {
                layer: 1,
                feature: 3,
                relation: None,
                entity: "C".into(),
                target: "D".into(),
                confidence: None,
                gate_vector_b64: Some("!!!!".into()), // invalid base64
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            },
        ]);
        let err = pv.try_apply_patch(patch).expect_err("corrupt vector");
        assert!(err.to_string().contains("gate_vector_b64"), "{err}");
        assert!(pv.overrides_meta.is_empty(), "no meta may be applied");
        assert!(pv.overrides_gate.is_empty(), "no gate may be applied");
        assert_eq!(pv.num_patches(), 0, "corrupt patch must not be recorded");
    }

    #[test]
    fn try_apply_patch_rejects_truncated_up_vector_on_update() {
        // Truncated base64 (non-multiple-of-4 length) in an Update op's
        // up vector: the op's down_meta must NOT be applied.
        let full = encode_gate_vector(&[1.0f32, -1.0]);
        let truncated = full[..full.len() - 1].to_string();
        let mut pv = empty_pv();
        let patch = make_patch(vec![PatchOp::Update {
            layer: 2,
            feature: 4,
            gate_vector_b64: None,
            up_vector_b64: Some(truncated),
            down_vector_b64: None,
            down_meta: Some(PatchDownMeta {
                top_token: "x".into(),
                top_token_id: 1,
                c_score: 0.5,
            }),
        }]);
        assert!(pv.try_apply_patch(patch).is_err());
        assert!(pv.overrides_meta.is_empty());
        assert!(pv.up_override_at(2, 4).is_none());
    }

    #[test]
    fn apply_patch_drops_corrupt_patch_wholesale() {
        // The infallible compatibility wrapper: corrupt patch is a
        // no-op, never a half-application.
        let mut pv = empty_pv();
        let patch = make_patch(vec![PatchOp::Insert {
            layer: 0,
            feature: 1,
            relation: None,
            entity: "A".into(),
            target: "B".into(),
            confidence: None,
            gate_vector_b64: Some("bad!".into()),
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]);
        pv.apply_patch(patch);
        assert!(pv.overrides_meta.is_empty());
        assert_eq!(pv.num_patches(), 0);
    }

    #[test]
    fn try_apply_patch_rejects_corrupt_knn_key() {
        let mut pv = empty_pv();
        let patch = make_patch(vec![PatchOp::InsertKnn {
            layer: 0,
            entity: "e".into(),
            relation: "r".into(),
            target: "t".into(),
            target_id: 1,
            confidence: None,
            key_vector_b64: "???".into(),
        }]);
        assert!(pv.try_apply_patch(patch).is_err());
        assert_eq!(pv.knn_store.len(), 0);
    }

    #[test]
    fn update_after_delete_clears_tombstone() {
        // Regression (review 2026-07-30 M6): PatchOp::Update on a
        // tombstoned slot must resurrect it, mirroring Insert. Before
        // the fix the tombstone survived, so feature_meta() (via the
        // Some(meta) override) and gate_knn() (tombstone filter)
        // disagreed about the same slot.
        let mut pv = empty_pv();
        pv.apply_patch(make_patch(vec![PatchOp::Delete {
            layer: 0,
            feature: 1,
            reason: None,
        }]));
        assert!(pv.deleted.contains(&(0, 1)));

        pv.apply_patch(make_patch(vec![PatchOp::Update {
            layer: 0,
            feature: 1,
            gate_vector_b64: Some(encode_gate_vector(&[1.0f32, 0.0])),
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(PatchDownMeta {
                top_token: "revived".into(),
                top_token_id: 7,
                c_score: 0.6,
            }),
        }]));

        assert!(
            !pv.deleted.contains(&(0, 1)),
            "Update must clear the tombstone"
        );
        // Both query paths agree the feature exists again.
        assert_eq!(pv.feature_meta(0, 1).unwrap().top_token, "revived");
        let q = ndarray::Array1::from_vec(vec![1.0_f32, 0.0]);
        let hits = pv.gate_knn(0, &q, 1);
        assert_eq!(hits.len(), 1, "got {hits:?}");
        assert_eq!(hits[0].0, 1, "resurrected feature must be visible to KNN");
    }

    #[test]
    fn update_without_meta_after_delete_drops_pinned_none_meta() {
        // Delete pins `overrides_meta[key] = None`; an Update carrying
        // no down_meta must not leave that pin behind after clearing
        // the tombstone, or feature_meta() would keep reporting None
        // for a slot gate_knn() now returns.
        let mut pv = empty_pv();
        pv.apply_patch(make_patch(vec![PatchOp::Delete {
            layer: 0,
            feature: 2,
            reason: None,
        }]));
        pv.apply_patch(make_patch(vec![PatchOp::Update {
            layer: 0,
            feature: 2,
            gate_vector_b64: Some(encode_gate_vector(&[0.0f32, 1.0])),
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]));
        assert!(!pv.deleted.contains(&(0, 2)));
        assert!(
            !matches!(pv.overrides_meta.get(&(0, 2)), Some(None)),
            "pinned-None meta override must be dropped on resurrection"
        );
    }

    #[test]
    fn apply_insert_knn_adds_to_knn_store() {
        let mut pv = empty_pv();
        let kv = encode_gate_vector(&[1.0f32, 0.0, 0.0]);
        let patch = make_patch(vec![PatchOp::InsertKnn {
            layer: 0,
            entity: "France".into(),
            relation: "capital".into(),
            target: "Paris".into(),
            target_id: 1234,
            confidence: Some(1.0),
            key_vector_b64: kv,
        }]);
        pv.apply_patch(patch);
        assert_eq!(pv.knn_store.len(), 1);
    }

    #[test]
    fn apply_delete_knn_removes_from_knn_store() {
        let mut pv = empty_pv();
        let kv = encode_gate_vector(&[1.0f32, 0.0, 0.0]);
        let insert = make_patch(vec![PatchOp::InsertKnn {
            layer: 0,
            entity: "France".into(),
            relation: "capital".into(),
            target: "Paris".into(),
            target_id: 1,
            confidence: None,
            key_vector_b64: kv,
        }]);
        let delete = make_patch(vec![PatchOp::DeleteKnn {
            entity: "France".into(),
        }]);
        pv.apply_patch(insert);
        assert_eq!(pv.knn_store.len(), 1);
        pv.apply_patch(delete);
        assert_eq!(pv.knn_store.len(), 0);
    }
}
