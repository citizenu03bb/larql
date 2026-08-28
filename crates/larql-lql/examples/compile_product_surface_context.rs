//! Compile an exact product-bounded surface manifest without loading weights.
//!
//! The source `.vlp` contributes only its checksum-bound terminal residual
//! adapter, logical slot, and complete target IDs. Its learned detector is
//! intentionally discarded. Supported prompts come from a separate frozen
//! JSON bank and are encoded through the active vindex tokenizer/architecture.

use std::collections::BTreeSet;
use std::error::Error;
use std::path::{Path, PathBuf};

use larql_inference::active_span::PatchSlotId;
use larql_inference::compiled_surface_context::{
    CompiledSurfaceContextManifest, CompiledSurfaceContextSpec,
};
use larql_inference::context_detector::{TerminalResidualAdapterSpec, TerminalResidualApplication};
use larql_vindex::patch::core::{base_checksum_from_manifest, decode_gate_vector};
use larql_vindex::{PatchOp, PatchResidualAdapterApplication, VindexPatch};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

type AnyError = Box<dyn Error>;

#[derive(Debug, Deserialize)]
struct SurfaceBank {
    protocol_id: String,
    supported_prompts: Vec<String>,
}

struct SourceAdapter {
    payload: PatchSlotId,
    target_ids: Vec<u32>,
    adapter: TerminalResidualAdapterSpec,
}

fn usage() -> &'static str {
    "usage: compile_product_surface_context <vindex> <adapter.vlp> <surfaces.json> <output.json>"
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_bytes(path: &Path) -> Result<Vec<u8>, AnyError> {
    Ok(std::fs::read(path)?)
}

fn extract_source_adapter(
    patch: &VindexPatch,
    hidden_size: usize,
) -> Result<SourceAdapter, AnyError> {
    let candidates: Vec<_> = patch
        .operations
        .iter()
        .filter_map(|operation| match operation {
            PatchOp::Context {
                layer,
                feature,
                target_ids,
                residual_adapter: Some(adapter),
                ..
            } => Some((*layer, *feature, target_ids, adapter)),
            _ => None,
        })
        .collect();
    if candidates.len() != 1 {
        return Err(format!(
            "source patch must contain exactly one contextual terminal adapter, found {}",
            candidates.len()
        )
        .into());
    }
    let (layer, feature, target_ids, adapter) = candidates[0];
    let application = match adapter.application {
        PatchResidualAdapterApplication::PrefillLastPositionV1 => {
            TerminalResidualApplication::PrefillLastPositionV1
        }
    };
    let delta = decode_gate_vector(&adapter.delta_vector_b64)?;
    if delta.len() != hidden_size {
        return Err(format!(
            "source terminal adapter has {} values, expected hidden size {hidden_size}",
            delta.len()
        )
        .into());
    }
    Ok(SourceAdapter {
        payload: PatchSlotId { layer, feature },
        target_ids: target_ids.clone(),
        adapter: TerminalResidualAdapterSpec {
            application,
            delta,
            calibration: adapter.calibration.clone(),
        },
    })
}

fn compile_prompt_suffixes<F>(prompts: &[String], mut encode: F) -> Result<Vec<Vec<u32>>, AnyError>
where
    F: FnMut(&str) -> Result<Vec<u32>, AnyError>,
{
    if prompts.is_empty() {
        return Err("surface bank contains no supported prompts".into());
    }
    let mut seen_prompts = BTreeSet::new();
    let mut seen_ids = BTreeSet::new();
    let mut suffixes = Vec::with_capacity(prompts.len());
    for (index, prompt) in prompts.iter().enumerate() {
        if prompt.is_empty() || prompt.trim() != prompt {
            return Err(format!(
                "supported prompt {index} must be non-empty and have no outer whitespace"
            )
            .into());
        }
        if !seen_prompts.insert(prompt) {
            return Err(format!("supported prompt {index} duplicates an earlier prompt").into());
        }
        let ids = encode(prompt)?;
        if ids.is_empty() {
            return Err(format!("supported prompt {index} encoded to no IDs").into());
        }
        if !seen_ids.insert(ids.clone()) {
            return Err(format!(
                "supported prompt {index} tokenizes identically to an earlier prompt"
            )
            .into());
        }
        suffixes.push(ids);
    }
    Ok(suffixes)
}

fn run(args: &[String]) -> Result<serde_json::Value, AnyError> {
    if args.len() != 5 {
        return Err(usage().into());
    }
    let vindex = PathBuf::from(&args[1]);
    let adapter_path = PathBuf::from(&args[2]);
    let surfaces_path = PathBuf::from(&args[3]);
    let output_path = PathBuf::from(&args[4]);

    let config = larql_vindex::load_vindex_config(&vindex)?;
    let arch = larql_vindex::arch_from_vindex_config(&config)
        .ok_or("vindex has no model_config for prompt compilation")?;
    let active_checksum = base_checksum_from_manifest(config.checksums.as_ref())
        .ok_or("vindex has no non-empty checksum manifest")?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex)?;

    let adapter_bytes = read_bytes(&adapter_path)?;
    let source: VindexPatch = serde_json::from_slice(&adapter_bytes)?;
    if source.base_checksum.as_deref() != Some(active_checksum.as_str()) {
        return Err("source adapter patch is not bound to the active vindex checksum".into());
    }
    let source_adapter = extract_source_adapter(&source, config.hidden_size)?;
    if source_adapter.payload.layer >= config.num_layers
        || source_adapter.payload.feature >= config.intermediate_size
    {
        return Err("source adapter logical slot is outside active model geometry".into());
    }

    let surfaces_bytes = read_bytes(&surfaces_path)?;
    let bank: SurfaceBank = serde_json::from_slice(&surfaces_bytes)?;
    if bank.protocol_id.trim().is_empty() {
        return Err("surface bank requires a protocol_id".into());
    }
    let prompt_suffixes = compile_prompt_suffixes(&bank.supported_prompts, |prompt| {
        larql_inference::encode_prompt(&tokenizer, arch.as_ref(), prompt)
            .map_err(|error| -> AnyError { Box::new(error) })
    })?;

    let spec = CompiledSurfaceContextSpec {
        base_checksum: active_checksum.clone(),
        payload: source_adapter.payload,
        target_ids: source_adapter.target_ids,
        prompt_suffixes,
        terminal_residual_adapter: source_adapter.adapter,
        provenance: Some(format!(
            "source_adapter_sha256={};surface_bank_sha256={}",
            sha256_bytes(&adapter_bytes),
            sha256_bytes(&surfaces_bytes)
        )),
    };
    spec.validate(&active_checksum, config.hidden_size, config.vocab_size)?;
    let manifest = CompiledSurfaceContextManifest::new(bank.protocol_id, vec![spec]);
    manifest.validate_header()?;

    let output = serde_json::to_vec_pretty(&manifest)?;
    let output_sha256 = sha256_bytes(&output);
    std::fs::write(&output_path, &output)?;

    let reparsed: CompiledSurfaceContextManifest =
        serde_json::from_slice(&std::fs::read(&output_path)?)?;
    reparsed.validate_header()?;
    for spec in &reparsed.specs {
        spec.validate(&active_checksum, config.hidden_size, config.vocab_size)?;
    }

    Ok(json!({
        "status": "compiled",
        "claim_label": manifest.claim_label,
        "protocol_id": manifest.protocol_id,
        "weights_loaded": false,
        "supported_prompts": bank.supported_prompts.len(),
        "target_ids": manifest.specs[0].target_ids,
        "logical_slot": {
            "layer": manifest.specs[0].payload.layer,
            "feature": manifest.specs[0].payload.feature,
        },
        "base_checksum": active_checksum,
        "output": output_path,
        "output_sha256": output_sha256,
    }))
}

fn main() -> Result<(), AnyError> {
    let args: Vec<String> = std::env::args().collect();
    println!("{}", serde_json::to_string_pretty(&run(&args)?)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_compiler_preserves_order_and_rejects_duplicates() {
        let prompts = vec!["first".to_string(), "second".to_string()];
        let suffixes = compile_prompt_suffixes(&prompts, |prompt| {
            Ok(prompt.bytes().map(u32::from).collect())
        })
        .unwrap();
        assert_eq!(suffixes[0][0], u32::from(b'f'));
        assert_eq!(suffixes[1][0], u32::from(b's'));

        let duplicates = vec!["same".to_string(), "same".to_string()];
        assert!(compile_prompt_suffixes(&duplicates, |_| Ok(vec![1])).is_err());
    }
}
