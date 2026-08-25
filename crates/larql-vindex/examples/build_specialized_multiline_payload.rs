//! Build a development-only K=2 contextual payload from a frozen K=1 patch.
//!
//! The second row keeps the source row's gate/up norms and exact down vector,
//! but uses a captured pre-FFN residual as its gate/up direction. The active
//! base overlay selects the second slot, so the tool never guesses a feature
//! coordinate or silently overwrites the source row.
//!
//! Usage:
//! `build_specialized_multiline_payload <vindex> <k1.vlp> <captures.json> \
//!    <address-prompt> <k2.vlp>`

use std::error::Error;
use std::path::{Path, PathBuf};

use larql_vindex::patch::core::{
    base_checksum_from_manifest, decode_gate_vector, encode_gate_vector,
};
use larql_vindex::{PatchOp, PatchedVindex, VindexPatch};
use serde_json::Value;

type AnyError = Box<dyn Error>;

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn scaled_unit(values: &[f32], target_norm: f32) -> Result<Vec<f32>, AnyError> {
    let norm = l2_norm(values);
    if values.is_empty() || !norm.is_finite() || norm <= 1e-8 {
        return Err("address residual must be finite, non-empty, and non-zero".into());
    }
    Ok(values
        .iter()
        .map(|value| value / norm * target_norm)
        .collect())
}

fn cosine(left: &[f32], right: &[f32]) -> Result<f32, AnyError> {
    if left.len() != right.len() {
        return Err(format!(
            "cosine dimension mismatch: {} vs {}",
            left.len(),
            right.len()
        )
        .into());
    }
    let denom = l2_norm(left) * l2_norm(right);
    if denom <= 1e-8 {
        return Err("cosine requires non-zero vectors".into());
    }
    Ok(left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>() / denom)
}

fn capture_residual(
    captures: &Value,
    address_prompt: &str,
    payload_layer: usize,
) -> Result<Vec<f32>, AnyError> {
    let rows = captures
        .as_array()
        .ok_or("capture file must contain a JSON array")?;
    let matches: Vec<_> = rows
        .iter()
        .filter(|row| row.get("prompt").and_then(Value::as_str) == Some(address_prompt))
        .collect();
    let [capture] = matches.as_slice() else {
        return Err(format!(
            "address prompt must match exactly one capture, found {}",
            matches.len()
        )
        .into());
    };
    let residuals = capture
        .get("residuals")
        .and_then(Value::as_array)
        .ok_or("matching capture has no residual array")?;
    let layer_rows: Vec<_> = residuals
        .iter()
        .filter(|row| {
            row.as_array()
                .and_then(|pair| pair.first())
                .and_then(Value::as_u64)
                == Some(payload_layer as u64)
        })
        .collect();
    let [layer_row] = layer_rows.as_slice() else {
        return Err(format!(
            "capture must contain exactly one L{payload_layer} residual, found {}",
            layer_rows.len()
        )
        .into());
    };
    let pair = layer_row.as_array().ok_or("residual row must be a pair")?;
    pair.get(1)
        .and_then(Value::as_array)
        .ok_or("residual vector is missing")?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|number| number as f32)
                .ok_or_else(|| "residual vector contains a non-number".into())
        })
        .collect()
}

fn source_ops(patch: &VindexPatch) -> Result<(PatchOp, PatchOp), AnyError> {
    let inserts: Vec<_> = patch
        .operations
        .iter()
        .filter(|op| matches!(op, PatchOp::Insert { .. }))
        .cloned()
        .collect();
    let contexts: Vec<_> = patch
        .operations
        .iter()
        .filter(|op| matches!(op, PatchOp::Context { .. }))
        .cloned()
        .collect();
    let [insert] = inserts.as_slice() else {
        return Err(format!("K=1 patch must contain one Insert, found {}", inserts.len()).into());
    };
    let [context] = contexts.as_slice() else {
        return Err(format!(
            "K=1 patch must contain one Context, found {}",
            contexts.len()
        )
        .into());
    };
    if insert.key() != context.key() {
        return Err("K=1 Insert and Context must target the same slot".into());
    }
    Ok((insert.clone(), context.clone()))
}

fn derive_patch(
    mut patch: VindexPatch,
    new_feature: usize,
    address_prompt: &str,
    address_residual: &[f32],
) -> Result<(VindexPatch, Value), AnyError> {
    let (source_insert, source_context) = source_ops(&patch)?;
    let PatchOp::Insert {
        layer,
        feature: source_feature,
        relation,
        entity,
        target,
        confidence,
        gate_vector_b64: Some(source_gate_b64),
        up_vector_b64: Some(source_up_b64),
        down_vector_b64: Some(source_down_b64),
        down_meta,
    } = source_insert
    else {
        return Err("source Insert must persist gate, up, and down vectors".into());
    };
    let PatchOp::Context {
        layer: context_layer,
        feature: context_feature,
        target_ids,
        detector,
        residual_adapter,
    } = source_context
    else {
        unreachable!("source_ops returned a non-context operation")
    };
    if layer != context_layer || source_feature != context_feature {
        return Err("source Insert and Context slot changed during derivation".into());
    }
    if new_feature == source_feature {
        return Err("derived feature must differ from the source feature".into());
    }

    let source_gate = decode_gate_vector(&source_gate_b64)?;
    let source_up = decode_gate_vector(&source_up_b64)?;
    let source_down = decode_gate_vector(&source_down_b64)?;
    if source_gate.len() != address_residual.len()
        || source_up.len() != address_residual.len()
        || source_down.len() != address_residual.len()
    {
        return Err(format!(
            "source/address dimensions differ: gate={}, up={}, down={}, address={}",
            source_gate.len(),
            source_up.len(),
            source_down.len(),
            address_residual.len()
        )
        .into());
    }

    let derived_gate = scaled_unit(address_residual, l2_norm(&source_gate))?;
    let derived_up = scaled_unit(address_residual, l2_norm(&source_up))?;
    let gate_address_cosine = cosine(&source_gate, address_residual)?;

    let derived_insert = PatchOp::Insert {
        layer,
        feature: new_feature,
        relation,
        entity,
        target,
        confidence,
        gate_vector_b64: Some(encode_gate_vector(&derived_gate)),
        up_vector_b64: Some(encode_gate_vector(&derived_up)),
        down_vector_b64: Some(source_down_b64),
        down_meta,
    };
    let derived_context = PatchOp::Context {
        layer,
        feature: new_feature,
        target_ids,
        detector,
        residual_adapter,
    };

    let first_context = patch
        .operations
        .iter()
        .position(|op| matches!(op, PatchOp::Context { .. }))
        .unwrap_or(patch.operations.len());
    patch.operations.insert(first_context, derived_insert);
    patch.operations.push(derived_context);
    patch.description = Some(format!(
        "Development-only K=2 surface-specialized payload; second row addressed by {address_prompt:?}"
    ));
    for tag in ["research-spike", "multi-row", "surface-specialized"] {
        if !patch.tags.iter().any(|existing| existing == tag) {
            patch.tags.push(tag.to_string());
        }
    }

    let report = serde_json::json!({
        "source_slot": {"layer": layer, "feature": source_feature},
        "derived_slot": {"layer": layer, "feature": new_feature},
        "address_prompt": address_prompt,
        "gate_address_cosine": gate_address_cosine,
        "source_gate_norm": l2_norm(&source_gate),
        "derived_gate_norm": l2_norm(&derived_gate),
        "source_up_norm": l2_norm(&source_up),
        "derived_up_norm": l2_norm(&derived_up),
        "source_down_norm": l2_norm(&source_down),
        "derived_down_is_exact_copy": true,
    });
    Ok((patch, report))
}

fn main() -> Result<(), AnyError> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err(format!(
            "usage: {} <vindex> <k1.vlp> <captures.json> <address-prompt> <k2.vlp>",
            args.first()
                .map(String::as_str)
                .unwrap_or("build_specialized_multiline_payload")
        )
        .into());
    }
    let vindex = PathBuf::from(&args[1]);
    let source_path = PathBuf::from(&args[2]);
    let capture_path = PathBuf::from(&args[3]);
    let address_prompt = &args[4];
    let output = PathBuf::from(&args[5]);

    let source_patch = VindexPatch::load(&source_path)?;
    let (source_insert, _) = source_ops(&source_patch)?;
    let (payload_layer, source_feature) = source_insert
        .key()
        .ok_or("source Insert has no payload slot")?;
    let captures: Value = serde_json::from_slice(&std::fs::read(&capture_path)?)?;
    let address_residual = capture_residual(&captures, address_prompt, payload_layer)?;

    let config = larql_vindex::load_vindex_config(&vindex)?;
    let active_checksum = base_checksum_from_manifest(config.checksums.as_ref());
    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let base = larql_vindex::VectorIndex::load_vindex_with_range(
        &vindex,
        &mut callbacks,
        Some((payload_layer, payload_layer + 1)),
    )?;
    let mut overlay = PatchedVindex::new(base);
    overlay.try_apply_patch_bound(source_patch.clone(), active_checksum.as_deref())?;
    let new_feature = overlay
        .find_free_feature(payload_layer)
        .ok_or_else(|| format!("no second free feature at L{payload_layer}"))?;
    if new_feature == source_feature {
        return Err("allocator returned the already claimed source feature".into());
    }

    let (derived, report) =
        derive_patch(source_patch, new_feature, address_prompt, &address_residual)?;
    derived.save(Path::new(&output))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_vindex::patch::core::PatchDownMeta;
    use larql_vindex::{PatchContextDetector, PatchContextRepresentation};

    fn fixture_patch() -> VindexPatch {
        let insert = PatchOp::Insert {
            layer: 3,
            feature: 7,
            relation: Some("ruler".into()),
            entity: "Lothlorien".into(),
            target: "Galadriel".into(),
            confidence: Some(0.9),
            gate_vector_b64: Some(encode_gate_vector(&[3.0, 0.0])),
            up_vector_b64: Some(encode_gate_vector(&[0.0, 2.0])),
            down_vector_b64: Some(encode_gate_vector(&[0.25, -0.5])),
            down_meta: Some(PatchDownMeta {
                top_token: "Galadriel".into(),
                top_token_id: 10,
                c_score: 0.9,
            }),
        };
        let context = PatchOp::Context {
            layer: 3,
            feature: 7,
            target_ids: vec![10, 11, 12],
            detector: PatchContextDetector {
                detector_layer: 1,
                representation: PatchContextRepresentation::MeanNoBos,
                center_vector_b64: encode_gate_vector(&[0.0, 0.0]),
                entity_direction_b64: encode_gate_vector(&[1.0, 0.0]),
                relation_direction_b64: encode_gate_vector(&[0.0, 1.0]),
                entity_threshold: 0.0,
                relation_threshold: 0.0,
                calibration: Some("fixture".into()),
            },
            residual_adapter: None,
        };
        VindexPatch {
            version: 1,
            base_model: "fixture".into(),
            base_checksum: Some("sha256-manifest:fixture".into()),
            created_at: "fixture".into(),
            description: None,
            author: None,
            tags: Vec::new(),
            operations: vec![insert, context],
        }
    }

    #[test]
    fn capture_requires_one_exact_prompt_and_layer() {
        let captures = serde_json::json!([{
            "prompt": "passive",
            "residuals": [[2, [9.0, 9.0]], [3, [1.0, 2.0]]]
        }]);
        assert_eq!(
            capture_residual(&captures, "passive", 3).unwrap(),
            [1.0, 2.0]
        );
        assert!(capture_residual(&captures, "missing", 3).is_err());
    }

    #[test]
    fn derived_row_preserves_scales_payload_and_context() {
        let source = fixture_patch();
        let (derived, report) = derive_patch(source.clone(), 8, "passive", &[1.0, 1.0]).unwrap();
        assert_eq!(derived.operations.len(), 4);
        let preserved_source_ops = derived.operations[..1]
            .iter()
            .chain(&derived.operations[2..3])
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            serde_json::to_value(&source.operations).unwrap(),
            serde_json::to_value(&preserved_source_ops).unwrap()
        );

        let inserts: Vec<_> = derived
            .operations
            .iter()
            .filter_map(|op| match op {
                PatchOp::Insert {
                    feature,
                    gate_vector_b64,
                    up_vector_b64,
                    down_vector_b64,
                    ..
                } => Some((feature, gate_vector_b64, up_vector_b64, down_vector_b64)),
                _ => None,
            })
            .collect();
        assert_eq!(inserts.len(), 2);
        assert_eq!(*inserts[1].0, 8);
        let source_gate = decode_gate_vector(inserts[0].1.as_ref().unwrap()).unwrap();
        let derived_gate = decode_gate_vector(inserts[1].1.as_ref().unwrap()).unwrap();
        let source_up = decode_gate_vector(inserts[0].2.as_ref().unwrap()).unwrap();
        let derived_up = decode_gate_vector(inserts[1].2.as_ref().unwrap()).unwrap();
        assert!((l2_norm(&source_gate) - l2_norm(&derived_gate)).abs() < 1e-6);
        assert!((l2_norm(&source_up) - l2_norm(&derived_up)).abs() < 1e-6);
        assert_eq!(inserts[0].3, inserts[1].3);

        let contexts: Vec<_> = derived
            .operations
            .iter()
            .filter_map(|op| match op {
                PatchOp::Context {
                    feature,
                    target_ids,
                    detector,
                    ..
                } => Some((feature, target_ids, detector)),
                _ => None,
            })
            .collect();
        assert_eq!(contexts.len(), 2);
        assert_eq!(*contexts[1].0, 8);
        assert_eq!(contexts[0].1, contexts[1].1);
        assert_eq!(contexts[0].2.detector_layer, contexts[1].2.detector_layer);
        assert_eq!(report["derived_down_is_exact_copy"], true);
    }
}
