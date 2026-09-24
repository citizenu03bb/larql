//! Weight-free preflight and bounded unpatched Q4K worker for NBCP-v1.
//! The checksummed Python controller owns fitting, durable evidence, and stage gates.
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::io::{BufRead, Write};
use std::path::Path;

use larql_inference::forward::{LayerHook, NoopHook, PreFfnPatch, PreFfnPatchHook, RecordHook};
use larql_inference::vindex::{predict_kquant_hidden_with_ffn_hooked, WalkFfn};
use larql_inference::{hidden_to_raw_logits, InferenceWeights};
use ndarray::Axis;
use serde::Deserialize;
use serde_json::{json, Value};

type ErrBox = Box<dyn Error>;
const SITE: usize = 21;
const WIDTH: usize = 2560;
const BINDING: &str =
    "sha256-manifest:c53cc460e0809529289375db6a90d84d038586d7cc005f19e401c8264066987a";

#[derive(Deserialize)]
struct Row {
    id: String,
    split: String,
    prompt: String,
    answer: Option<String>,
    alternative: Option<String>,
}
#[derive(Clone, Deserialize)]
struct Job {
    split: String,
    recipient: String,
    arm: String,
}
#[derive(Deserialize)]
struct Request {
    ordinal: usize,
    recipient: String,
    arm: String,
    replacement: Option<Vec<f64>>,
}

fn emit(value: &Value) -> Result<(), ErrBox> {
    let mut out = std::io::stdout().lock();
    write!(out, "NBCP\t")?;
    serde_json::to_writer(&mut out, value)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}

fn checked_patch(row: &[f64], position: usize, width: usize) -> Result<PreFfnPatch, ErrBox> {
    if row.len() != width || row.iter().any(|v| !v.is_finite()) {
        return Err("invalid replacement shape or nonfinite data".into());
    }
    let donor_row: Vec<f32> = row.iter().map(|&v| v as f32).collect();
    if donor_row.iter().any(|v| !v.is_finite()) {
        return Err("replacement overflowed f32".into());
    }
    Ok(PreFfnPatch {
        layer: SITE,
        recipient_position: position,
        donor_row,
    })
}

fn check_request(request: &Request, job: &Job, ordinal: usize) -> Result<(), ErrBox> {
    if ordinal >= 184
        || request.ordinal != ordinal
        || request.recipient != job.recipient
        || request.arm != job.arm
    {
        return Err("request deviates from frozen forward ledger".into());
    }
    let baseline = matches!(
        request.arm.as_str(),
        "baseline_capture" | "hook_disabled_replay"
    );
    if baseline == request.replacement.is_some() {
        return Err("replacement presence disagrees with arm".into());
    }
    Ok(())
}

fn forward(
    weights: &mut larql_models::ModelWeights,
    index: &larql_vindex::VectorIndex,
    ids: &[u32],
    hook: &mut dyn LayerHook,
) -> Result<Vec<f32>, ErrBox> {
    // Same field-disjoint alias as the existing Q4K causal research runner:
    // the FFN reads immutable weights while attention mutates its scratch map.
    let weights_ref: &larql_models::ModelWeights = unsafe { &*(weights as *const _) };
    let ffn = WalkFfn::new_unlimited(weights_ref, index);
    let h = predict_kquant_hidden_with_ffn_hooked(weights, ids, index, &ffn, hook)?;
    let last = h.row(h.nrows() - 1).to_owned().insert_axis(Axis(0));
    let logits = hidden_to_raw_logits(weights, &last);
    if logits.len() != 262144 || logits.iter().any(|v| !v.is_finite()) {
        return Err("invalid full logits".into());
    }
    Ok(logits)
}

fn main() -> Result<(), ErrBox> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 6
        || !["preflight", "serve"].contains(&args[0].as_str())
        || !["discovery", "confirmation"].contains(&args[1].as_str())
    {
        return Err("usage: natural_binding_causal_pilot preflight|serve discovery|confirmation VINDEX CORPUS TOKENS LEDGER".into());
    }
    let (mode, stage) = (&args[0], &args[1]);
    let path = Path::new(&args[2]);
    let config = larql_vindex::load_vindex_config(path)?;
    if config.num_layers != 42 || config.hidden_size != WIDTH || config.vocab_size != 262144 {
        return Err("frozen model geometry mismatch".into());
    }
    if larql_vindex::patch::base_checksum_from_manifest(config.checksums.as_ref()).as_deref()
        != Some(BINDING)
    {
        return Err("base artifact binding mismatch".into());
    }
    let arch =
        larql_vindex::arch_from_vindex_config(&config).ok_or("missing architecture metadata")?;
    if arch.final_logit_softcapping() != Some(30.0) || arch.bos_token_id() != Some(2) {
        return Err("softcap or BOS differs from frozen design".into());
    }
    let tokenizer = larql_vindex::load_vindex_tokenizer(path)?;
    let rows: Vec<Row> = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    let tokens: Value = serde_json::from_slice(&std::fs::read(&args[4])?)?;
    let ledger: Vec<Job> = serde_json::from_slice(&std::fs::read(&args[5])?)?;
    let jobs: Vec<_> = ledger.into_iter().filter(|j| &j.split == stage).collect();
    if jobs.len() != 184 {
        return Err("invalid stage budget".into());
    }
    let mut prompts = BTreeMap::new();
    let mut evidence = Vec::new();
    // Check all authored text against the frozen Python preflight without inference.
    for row in &rows {
        // Same implementation path as LQL's crate-private encode_vindex_prompt.
        let ids = larql_inference::encode_prompt(&tokenizer, arch.as_ref(), &row.prompt)?;
        let old = tokens["rows"]
            .as_array()
            .ok_or("missing token rows")?
            .iter()
            .find(|x| x["id"] == row.id)
            .ok_or("missing token row")?;
        if serde_json::to_value(&ids)? != old["prompt_ids_with_bos"]
            || ids.first() != Some(&2)
            || ids.iter().filter(|&&x| x == 2).count() != 1
        {
            return Err(format!("Rust/Python prompt token mismatch: {}", row.id).into());
        }
        for (answer, field) in [
            (&row.answer, "answer_id"),
            (&row.alternative, "alternative_id"),
        ] {
            if let Some(answer) = answer {
                let full = larql_inference::encode_prompt(
                    &tokenizer,
                    arch.as_ref(),
                    &format!("{}{answer}", row.prompt),
                )?;
                if full.len() != ids.len() + 1
                    || full[..ids.len()] != ids
                    || Some(full[ids.len()] as u64) != old[field].as_u64()
                {
                    return Err(format!("Rust/Python answer token mismatch: {}", row.id).into());
                }
            }
        }
        evidence.push(json!({"id":row.id,"prompt_ids":ids,"position":ids.len()-1}));
        if &row.split == stage {
            prompts.insert(row.id.clone(), ids);
        }
    }
    if prompts.len() != 20 {
        return Err("invalid stage prompt count".into());
    }
    let unique: BTreeSet<_> = jobs.iter().map(|j| (&j.recipient, &j.arm)).collect();
    if unique.len() != 184 || jobs.iter().any(|j| !prompts.contains_key(&j.recipient)) {
        return Err("duplicate or unknown ledger jobs".into());
    }
    emit(
        &json!({"type":"preflight","status":"pass","stage":stage,"rows":evidence,"model_forwards":0,"binding":BINDING}),
    )?;
    if mode == "preflight" {
        return Ok(());
    }
    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let mut inference = InferenceWeights::load(path, &config, &mut callbacks)?;
    let (weights, index) = inference
        .as_quantised_parts_mut()
        .ok_or("native Q4K required")?;
    emit(&json!({"type":"ready","stage":stage}))?;
    for (ordinal, line) in std::io::stdin().lock().lines().enumerate() {
        let request: Request = serde_json::from_str(&line?)?;
        let job = jobs.get(ordinal).ok_or("stage budget exhausted")?;
        check_request(&request, job, ordinal)?;
        let ids = &prompts[&request.recipient];
        let mut capture = None;
        let mut applied = Vec::new();
        let mut rejected = Vec::new();
        let mut replacement_f32 = None;
        let logits = if request.arm == "baseline_capture" {
            let mut hook = RecordHook::for_layers([SITE]);
            let logits = forward(weights, index, ids, &mut hook)?;
            let h = hook.post_attention.get(&SITE).ok_or("missing capture")?;
            if h.nrows() != ids.len() || h.ncols() != WIDTH || h.iter().any(|v| !v.is_finite()) {
                return Err("invalid pre-FFN capture".into());
            }
            capture = Some(h.row(ids.len() - 1).to_vec());
            logits
        } else if request.arm == "hook_disabled_replay" {
            forward(weights, index, ids, &mut NoopHook)?
        } else {
            let patch = checked_patch(
                request.replacement.as_ref().ok_or("missing replacement")?,
                ids.len() - 1,
                WIDTH,
            )?;
            replacement_f32 = Some(patch.donor_row.clone());
            let patches = [patch];
            let mut hook = PreFfnPatchHook::new(&patches);
            let logits = forward(weights, index, ids, &mut hook)?;
            applied = hook.applied;
            rejected = hook.rejected;
            if applied != vec![(SITE, ids.len() - 1)] || !rejected.is_empty() {
                return Err("hook accounting invalid".into());
            }
            logits
        };
        emit(
            &json!({"type":"forward","ordinal":ordinal,"recipient":request.recipient,"arm":request.arm,
                     "logits":logits,"capture":capture,"applied":applied,"rejected":rejected,"replacement_f32":replacement_f32}),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn actual_hook_applies_reference_row_only_at_requested_coordinate() {
        // Python reference: u=(0.6,0.8,0), r=(1,2,3), d=(-2,5,7).
        let row = [1.36_f64, 2.48, 3.0];
        let patch = checked_patch(&row, 1, 3).unwrap();
        assert_eq!(patch.donor_row, vec![1.36_f32, 2.48, 3.0]);
        let patches = [patch];
        let mut hook = PreFfnPatchHook::new(&patches);
        let original = array![[7., 8., 9.], [1., 2., 3.], [4., 5., 6.]];
        let mut h = original.clone();
        hook.on_post_attention(SITE - 1, &mut h);
        hook.on_post_layer(SITE, &mut h);
        assert_eq!(h, original);
        hook.on_post_attention(SITE, &mut h);
        assert_eq!(h.row(0), original.row(0));
        assert_eq!(h.row(2), original.row(2));
        assert_eq!(h.row(1).to_vec(), vec![1.36_f32, 2.48, 3.0]);
        assert_eq!(hook.applied, vec![(SITE, 1)]);
        assert!(hook.rejected.is_empty());
    }
    #[test]
    fn bad_replacements_fail_before_hook_and_invalid_positions_are_rejected() {
        assert!(checked_patch(&[1.], 0, 2).is_err());
        assert!(checked_patch(&[f64::NAN], 0, 1).is_err());
        assert!(checked_patch(&[f64::INFINITY], 0, 1).is_err());
        assert!(checked_patch(&[f64::MAX], 0, 1).is_err());
        let patches = [checked_patch(&[1., 2., 3.], 8, 3).unwrap()];
        let mut hook = PreFfnPatchHook::new(&patches);
        let mut h = array![[4., 5., 6.]];
        hook.on_post_attention(SITE, &mut h);
        assert_eq!(h, array![[4., 5., 6.]]);
        assert_eq!(hook.rejected, vec![(SITE, 8)]);
        assert!(hook.applied.is_empty());
    }
    #[test]
    fn rejects_replay_and_over_budget_requests() {
        let job = Job {
            split: "discovery".into(),
            recipient: "r".into(),
            arm: "self".into(),
        };
        let req = Request {
            ordinal: 4,
            recipient: "r".into(),
            arm: "self".into(),
            replacement: Some(vec![0.]),
        };
        assert!(check_request(&req, &job, 4).is_ok());
        assert!(check_request(&req, &job, 5).is_err());
        assert!(check_request(&req, &job, 184).is_err());
    }
}
