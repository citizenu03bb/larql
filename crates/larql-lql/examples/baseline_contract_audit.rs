//! Bounded, unpatched, full-prefix greedy diagnostic. No model load in preflight.
use larql_inference::{
    forward::NoopHook,
    hidden_to_raw_logits,
    vindex::{predict_kquant_hidden_with_ffn_hooked, WalkFfn},
    InferenceWeights,
};
use ndarray::Axis;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
};
type E = Box<dyn Error>;
#[derive(Deserialize)]
struct Case {
    id: String,
    rendered: String,
    ids: Vec<u32>,
    original_logit_digest: Option<String>,
}
#[derive(Deserialize)]
struct Design {
    max_new_tokens: usize,
    eos_ids: Vec<u32>,
    cases: Vec<Case>,
}
fn fresh(path: &Path, bytes: &[u8]) -> Result<(), E> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}
fn event(f: &mut File, x: Value) -> Result<(), E> {
    serde_json::to_writer(&mut *f, &x)?;
    writeln!(f)?;
    f.sync_all()?;
    println!("AUDIT\t{x}");
    std::io::stdout().flush()?;
    Ok(())
}
fn top1(logits: &[f32]) -> Result<u32, E> {
    if logits.len() != 262144 || logits.iter().any(|v| !v.is_finite()) {
        return Err("invalid logits".into());
    }
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    Ok(best as u32)
}
fn main() -> Result<(), E> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 || !["preflight", "run"].contains(&args[0].as_str()) {
        return Err("usage: baseline_contract_audit preflight|run VINDEX CASES OUTPUT".into());
    }
    let path = Path::new(&args[1]);
    let config = larql_vindex::load_vindex_config(path)?;
    let arch = larql_vindex::arch_from_vindex_config(&config).ok_or("missing architecture")?;
    if config.num_layers != 42
        || config.hidden_size != 2560
        || config.vocab_size != 262144
        || arch.final_logit_softcapping() != Some(30.0)
        || larql_vindex::patch::base_checksum_from_manifest(config.checksums.as_ref()).as_deref()
            != Some(
                "sha256-manifest:c53cc460e0809529289375db6a90d84d038586d7cc005f19e401c8264066987a",
            )
    {
        return Err("model mismatch".into());
    }
    let design: Design = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    if design.max_new_tokens != 8 || design.eos_ids != [1, 106] || design.cases.len() != 4 {
        return Err("invalid budget".into());
    }
    let tok = larql_vindex::load_vindex_tokenizer(path)?;
    for c in &design.cases {
        let ids = tok
            .encode(c.rendered.as_str(), false)
            .map_err(|e| format!("tokenizer: {e}"))?
            .get_ids()
            .to_vec();
        if ids != c.ids
            || ids.first() != Some(&2)
            || ids.iter().filter(|&&t| t == 2).count() != 1
            || ids.len() >= 128
        {
            return Err("prompt parity failed".into());
        }
    }
    println!(
        "AUDIT\t{}",
        json!({"type":"preflight","cases":4,"budget":32,"model_forwards":0})
    );
    if args[0] == "preflight" {
        return Ok(());
    }
    let out = Path::new(&args[3]);
    std::fs::create_dir(out)?;
    let mut events = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("events.jsonl"))?;
    event(&mut events, json!({"type":"load_start"}))?;
    let mut inference =
        InferenceWeights::load(path, &config, &mut larql_vindex::SilentLoadCallbacks)?;
    let (weights, index) = inference.as_quantised_parts_mut().ok_or("Q4K required")?;
    let mut ordinal = 0;
    let mut results = Vec::new();
    for c in &design.cases {
        let mut ids = c.ids.clone();
        let mut generated = Vec::new();
        let mut stop = "length";
        for step in 0..design.max_new_tokens {
            if ordinal >= 32 {
                return Err("budget exceeded".into());
            }
            event(
                &mut events,
                json!({"type":"attempt","ordinal":ordinal,"case":c.id,"step":step,"input_ids":ids}),
            )?;
            // Existing field-disjoint Q4K path: FFN weights are read-only; attention scratch is mutable.
            let weights_ref: &larql_models::ModelWeights = unsafe { &*(weights as *const _) };
            let ffn = WalkFfn::new_unlimited(weights_ref, index);
            let hidden =
                predict_kquant_hidden_with_ffn_hooked(weights, &ids, index, &ffn, &mut NoopHook)?;
            let last = hidden
                .row(hidden.nrows() - 1)
                .to_owned()
                .insert_axis(Axis(0));
            let logits = hidden_to_raw_logits(weights, &last);
            let next = top1(&logits)?;
            let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
            let digest = format!("{:x}", Sha256::digest(&bytes));
            let filename = format!("{ordinal:03}-logits.f32le");
            fresh(&out.join(&filename), &bytes)?;
            event(
                &mut events,
                json!({"type":"complete","ordinal":ordinal,"case":c.id,"step":step,"top1_id":next,"digest":digest,"artifact":filename}),
            )?;
            ordinal += 1;
            if step == 0 {
                if let Some(ref expected) = c.original_logit_digest {
                    if *expected != digest {
                        return Err("raw historical replay mismatch: stop all inference".into());
                    }
                }
            }
            generated.push(next);
            if design.eos_ids.contains(&next) {
                stop = "eos";
                break;
            }
            ids.push(next);
        }
        let decoded = tok
            .decode(&generated, false)
            .map_err(|e| format!("decode: {e}"))?;
        let result = json!({"case":c.id,"generated_ids":generated,"text":decoded,"stop":stop});
        event(&mut events, json!({"type":"case_complete","result":result}))?;
        results.push(result);
    }
    fresh(
        &out.join("results.json"),
        &serde_json::to_vec_pretty(
            &json!({"status":"complete","forwards":ordinal,"results":results}),
        )?,
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_nonfinite_and_resolves_ties_to_first() {
        assert!(top1(&[0.0]).is_err());
        let mut v = vec![0.0; 262144];
        v[12] = 1.0;
        v[13] = 1.0;
        assert_eq!(top1(&v).unwrap(), 12);
        v[100] = f32::NAN;
        assert!(top1(&v).is_err());
    }
    #[test]
    fn evidence_cannot_be_overwritten() {
        let p = std::env::temp_dir().join(format!("baseline-audit-test-{}", std::process::id()));
        fresh(&p, b"first").unwrap();
        assert!(fresh(&p, b"second").is_err());
        assert_eq!(std::fs::read(&p).unwrap(), b"first");
        std::fs::remove_file(p).unwrap();
    }
}
