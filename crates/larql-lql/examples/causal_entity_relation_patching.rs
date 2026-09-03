//! Frozen exploratory runner for causal entity-relation activation patching.
//!
//! Protocol: `docs/research-causal-entity-relation-patching-preregistration.md`.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::path::{Path, PathBuf};

use larql_inference::forward::{LayerHook, NoopHook, PreFfnPatch, PreFfnPatchHook, RecordHook};
use larql_inference::vindex::{predict_kquant_hidden_with_ffn_hooked, WalkFfn};
use larql_inference::{hidden_to_raw_logits, InferenceWeights};
use larql_lql::{parse, Session};
use ndarray::Axis;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

type AnyError = Box<dyn Error>;

const SITES: [usize; 4] = [11, 13, 12, 33];
const CELLS: [&str; 4] = ["11", "10", "01", "00"];
const TARGET_ID: usize = 107_545;
const DISTRACTOR_ID: usize = 506;
const EXPECTED_FORWARDS: usize = 240;

#[derive(Clone, Deserialize)]
struct FactSpec {
    entity: String,
    relation: String,
    target: String,
    expected_tokens: Option<usize>,
    #[serde(default)]
    paraphrases: Vec<String>,
}

#[derive(Clone, Deserialize)]
struct ControlSpec {
    name: String,
    prompt: String,
    cell: String,
    fold: usize,
}

#[derive(Deserialize)]
struct SuiteSpec {
    facts: Vec<FactSpec>,
    controls: Vec<ControlSpec>,
}

#[derive(Clone)]
struct PromptSpec {
    name: String,
    prompt: String,
    cell: String,
    fold: usize,
    prompt_ids: Vec<u32>,
}

#[derive(Clone)]
struct Baseline {
    prompt: PromptSpec,
    donor_rows: BTreeMap<usize, Vec<f32>>,
    logits: Vec<f32>,
    digest: String,
}

#[derive(Serialize)]
struct BaselineOutput {
    fold: usize,
    cell: String,
    name: String,
    prompt: String,
    prompt_ids: Vec<u32>,
    target_logit: f32,
    distractor_logit: f32,
    gap: f32,
    top1_id: usize,
    full_logits_sha256: String,
    noop_full_logits_sha256: String,
    noop_digest_identical: bool,
}

#[derive(Serialize)]
struct ArmOutput {
    site: usize,
    fold: usize,
    arm: &'static str,
    donor_fold: usize,
    donor_cell: String,
    donor_name: String,
    recipient_cell: String,
    recipient_name: String,
    donor_position: usize,
    recipient_position: usize,
    applied: Vec<(usize, usize)>,
    rejected: Vec<(usize, usize)>,
    target_logit: f32,
    distractor_logit: f32,
    gap: f32,
    delta_target: f32,
    delta_distractor: f32,
    delta_gap: f32,
    top1_id: usize,
    full_logits_sha256: String,
}

#[derive(Serialize)]
struct RunOutput {
    schema: &'static str,
    vindex: String,
    suite: String,
    source_patch: String,
    sites: [usize; 4],
    target_id: usize,
    distractor_id: usize,
    baseline_forwards: usize,
    noop_forwards: usize,
    patched_forwards: usize,
    total_forwards: usize,
    baselines: Vec<BaselineOutput>,
    arms: Vec<ArmOutput>,
}

struct Args {
    vindex: PathBuf,
    suite: PathBuf,
    source_patch: PathBuf,
    output: PathBuf,
}

fn parse_args() -> Result<Args, AnyError> {
    let mut args = std::env::args().skip(1);
    let vindex = args.next().ok_or("missing vindex path")?.into();
    let mut suite = None;
    let mut source_patch = None;
    let mut output = None;
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--suite" => suite = Some(value.into()),
            "--apply-patch" => source_patch = Some(value.into()),
            "--output" => output = Some(value.into()),
            _ => return Err(format!("unknown argument {flag}").into()),
        }
    }
    Ok(Args {
        vindex,
        suite: suite.ok_or("missing --suite")?,
        source_patch: source_patch.ok_or("missing --apply-patch")?,
        output: output.ok_or("missing --output")?,
    })
}

fn canonical_prompt(fact: &FactSpec) -> String {
    format!(
        "The {} of {} is",
        fact.relation.replace(['-', '_'], " "),
        fact.entity
    )
}

fn encode_prompt(
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    weights: &InferenceWeights,
    prompt: &str,
) -> Result<Vec<u32>, AnyError> {
    Ok(larql_inference::encode_prompt(
        tokenizer,
        weights.as_weights().arch.as_ref(),
        prompt,
    )?)
}

fn single_token_id(
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    surface: &str,
) -> Result<u32, AnyError> {
    let encoded = tokenizer
        .encode(surface, false)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let ids = encoded.get_ids();
    if ids.len() != 1 {
        return Err(format!("{surface:?} must encode to one ID, got {ids:?}").into());
    }
    Ok(ids[0])
}

fn prompt_specs(
    suite: &SuiteSpec,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    weights: &InferenceWeights,
) -> Result<Vec<PromptSpec>, AnyError> {
    let [fact] = suite.facts.as_slice() else {
        return Err(format!(
            "suite must contain exactly one fact, got {}",
            suite.facts.len()
        )
        .into());
    };
    if fact.target != "Nora Veldt" || fact.expected_tokens != Some(4) {
        return Err("suite target must remain Nora Veldt with T=4".into());
    }
    if fact.paraphrases.len() != 5 {
        return Err("suite must contain canonical plus five positive constructions".into());
    }

    let mut prompts = Vec::with_capacity(24);
    for (fold, prompt) in std::iter::once(canonical_prompt(fact))
        .chain(fact.paraphrases.iter().cloned())
        .enumerate()
    {
        prompts.push(PromptSpec {
            name: if fold == 0 {
                "canonical".into()
            } else {
                format!("paraphrase-{fold}")
            },
            prompt_ids: encode_prompt(tokenizer, weights, &prompt)?,
            prompt,
            cell: "11".into(),
            fold,
        });
    }
    for control in &suite.controls {
        prompts.push(PromptSpec {
            name: control.name.clone(),
            prompt_ids: encode_prompt(tokenizer, weights, &control.prompt)?,
            prompt: control.prompt.clone(),
            cell: control.cell.clone(),
            fold: control.fold,
        });
    }

    if prompts.len() != 24 {
        return Err(format!("suite must expand to 24 prompts, got {}", prompts.len()).into());
    }
    let unique_prompts: BTreeSet<_> = prompts
        .iter()
        .map(|prompt| prompt.prompt.as_str())
        .collect();
    if unique_prompts.len() != prompts.len() {
        return Err("suite prompts must be unique".into());
    }
    for fold in 0..6 {
        for cell in CELLS {
            let count = prompts
                .iter()
                .filter(|prompt| prompt.fold == fold && prompt.cell == cell)
                .count();
            if count != 1 {
                return Err(format!("fold {fold} cell {cell} occurs {count} times").into());
            }
        }
    }
    if prompts.iter().any(|prompt| prompt.prompt_ids.is_empty()) {
        return Err("every prompt must encode to at least one ID".into());
    }
    prompts.sort_by_key(|prompt| {
        let cell_order = CELLS.iter().position(|cell| *cell == prompt.cell).unwrap();
        (prompt.fold, cell_order)
    });
    Ok(prompts)
}

fn open_patched_session(vindex: &Path, patch: &Path) -> Result<Session, AnyError> {
    let mut session = Session::new();
    let use_stmt = parse(&format!(r#"USE "{}";"#, vindex.display()))?;
    for line in session.execute(&use_stmt)? {
        eprintln!("{line}");
    }
    let apply_stmt = parse(&format!(r#"APPLY PATCH "{}";"#, patch.display()))?;
    for line in session.execute(&apply_stmt)? {
        eprintln!("{line}");
    }
    Ok(session)
}

fn forward_with_hook(
    weights: &mut larql_models::ModelWeights,
    source_index: &larql_vindex::VectorIndex,
    patched: &larql_vindex::PatchedVindex,
    prompt_ids: &[u32],
    hook: &mut dyn LayerHook,
) -> Result<Vec<f32>, AnyError> {
    // Same field-disjoint alias used by `InferenceWeights::infer_patched_q4k`:
    // WalkFfn reads architecture/vectors while Q4K attention temporarily
    // mutates only the tensor scratch map.
    let weights_ref: &larql_models::ModelWeights = unsafe { &*(weights as *const _) };
    let ffn = WalkFfn::new_unlimited(weights_ref, patched);
    let hidden =
        predict_kquant_hidden_with_ffn_hooked(weights, prompt_ids, source_index, &ffn, hook)?;
    let last = hidden
        .row(hidden.nrows() - 1)
        .to_owned()
        .insert_axis(Axis(0));
    let logits = hidden_to_raw_logits(weights, &last);
    if logits.len() <= TARGET_ID || logits.len() <= DISTRACTOR_ID {
        return Err(format!(
            "vocabulary too small for frozen token IDs: {}",
            logits.len()
        )
        .into());
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("forward produced non-finite logits".into());
    }
    Ok(logits)
}

fn logits_digest(logits: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in logits {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn top1_id(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn key(fold: usize, cell: &str) -> (usize, String) {
    (fold, cell.to_string())
}

fn baseline_output(baseline: &Baseline, noop_digest: String) -> BaselineOutput {
    let target = baseline.logits[TARGET_ID];
    let distractor = baseline.logits[DISTRACTOR_ID];
    BaselineOutput {
        fold: baseline.prompt.fold,
        cell: baseline.prompt.cell.clone(),
        name: baseline.prompt.name.clone(),
        prompt: baseline.prompt.prompt.clone(),
        prompt_ids: baseline.prompt.prompt_ids.clone(),
        target_logit: target,
        distractor_logit: distractor,
        gap: target - distractor,
        top1_id: top1_id(&baseline.logits),
        full_logits_sha256: baseline.digest.clone(),
        noop_digest_identical: baseline.digest == noop_digest,
        noop_full_logits_sha256: noop_digest,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    weights: &mut larql_models::ModelWeights,
    source_index: &larql_vindex::VectorIndex,
    patched: &larql_vindex::PatchedVindex,
    site: usize,
    fold: usize,
    arm: &'static str,
    donor: &Baseline,
    recipient: &Baseline,
) -> Result<ArmOutput, AnyError> {
    let donor_position = donor.prompt.prompt_ids.len() - 1;
    let recipient_position = recipient.prompt.prompt_ids.len() - 1;
    let donor_row = donor
        .donor_rows
        .get(&site)
        .ok_or_else(|| format!("donor {} missing L{site}", donor.prompt.name))?
        .clone();
    let patches = [PreFfnPatch {
        layer: site,
        recipient_position,
        donor_row,
    }];
    let mut hook = PreFfnPatchHook::new(&patches);
    let logits = forward_with_hook(
        weights,
        source_index,
        patched,
        &recipient.prompt.prompt_ids,
        &mut hook,
    )?;
    if hook.applied != vec![(site, recipient_position)] || !hook.rejected.is_empty() {
        return Err(format!(
            "hook accounting failed for L{site} {arm}: applied={:?} rejected={:?}",
            hook.applied, hook.rejected
        )
        .into());
    }

    let target = logits[TARGET_ID];
    let distractor = logits[DISTRACTOR_ID];
    let baseline_target = recipient.logits[TARGET_ID];
    let baseline_distractor = recipient.logits[DISTRACTOR_ID];
    Ok(ArmOutput {
        site,
        fold,
        arm,
        donor_fold: donor.prompt.fold,
        donor_cell: donor.prompt.cell.clone(),
        donor_name: donor.prompt.name.clone(),
        recipient_cell: recipient.prompt.cell.clone(),
        recipient_name: recipient.prompt.name.clone(),
        donor_position,
        recipient_position,
        applied: hook.applied,
        rejected: hook.rejected,
        target_logit: target,
        distractor_logit: distractor,
        gap: target - distractor,
        delta_target: target - baseline_target,
        delta_distractor: distractor - baseline_distractor,
        delta_gap: (target - distractor) - (baseline_target - baseline_distractor),
        top1_id: top1_id(&logits),
        full_logits_sha256: logits_digest(&logits),
    })
}

fn main() -> Result<(), AnyError> {
    let args = parse_args()?;
    let suite: SuiteSpec = serde_json::from_slice(&std::fs::read(&args.suite)?)?;
    let config = larql_vindex::load_vindex_config(&args.vindex)?;
    if config.num_layers != 42 || config.hidden_size != 2560 {
        return Err(format!(
            "frozen architecture requires 42 layers and hidden 2560, got {} and {}",
            config.num_layers, config.hidden_size
        )
        .into());
    }
    let tokenizer = larql_vindex::load_vindex_tokenizer(&args.vindex)?;
    if single_token_id(&tokenizer, " Nora")? as usize != TARGET_ID
        || single_token_id(&tokenizer, " the")? as usize != DISTRACTOR_ID
    {
        return Err("frozen target or distractor token ID changed".into());
    }

    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let mut inference_weights = InferenceWeights::load(&args.vindex, &config, &mut callbacks)?;
    if !inference_weights.is_quantised() {
        return Err("frozen runner requires native quantised storage".into());
    }
    let prompts = prompt_specs(&suite, &tokenizer, &inference_weights)?;
    let session = open_patched_session(&args.vindex, &args.source_patch)?;
    let patched = session
        .patched_overlay()
        .ok_or("session has no patched vindex overlay")?;
    if patched.context_detectors().len() != 0 {
        return Err("source patch must remain unconditional for this exploratory run".into());
    }
    let (weights, source_index) = inference_weights
        .as_quantised_parts_mut()
        .ok_or("quantised parts unavailable")?;

    let mut forward_count = 0usize;
    let mut baselines = BTreeMap::new();
    for prompt in prompts {
        let mut record = RecordHook::for_layers(SITES);
        let logits = forward_with_hook(
            weights,
            source_index,
            patched,
            &prompt.prompt_ids,
            &mut record,
        )?;
        forward_count += 1;
        let mut donor_rows = BTreeMap::new();
        for site in SITES {
            let matrix = record
                .post_attention
                .get(&site)
                .ok_or_else(|| format!("{} missing pre-FFN L{site}", prompt.name))?;
            if matrix.nrows() != prompt.prompt_ids.len() || matrix.ncols() != 2560 {
                return Err(format!("{} has invalid L{site} capture shape", prompt.name).into());
            }
            donor_rows.insert(site, matrix.row(matrix.nrows() - 1).to_vec());
        }
        let digest = logits_digest(&logits);
        eprintln!(
            "baseline {}/24 fold={} cell={} name={}",
            forward_count, prompt.fold, prompt.cell, prompt.name
        );
        baselines.insert(
            key(prompt.fold, &prompt.cell),
            Baseline {
                prompt,
                donor_rows,
                logits,
                digest,
            },
        );
    }

    let mut baseline_outputs = Vec::with_capacity(24);
    for baseline in baselines.values() {
        let mut noop = NoopHook;
        let logits = forward_with_hook(
            weights,
            source_index,
            patched,
            &baseline.prompt.prompt_ids,
            &mut noop,
        )?;
        forward_count += 1;
        let digest = logits_digest(&logits);
        if digest != baseline.digest {
            return Err(format!(
                "hook-disabled digest mismatch for {}: {} != {}",
                baseline.prompt.name, digest, baseline.digest
            )
            .into());
        }
        baseline_outputs.push(baseline_output(baseline, digest));
    }

    let mut arms = Vec::with_capacity(192);
    for site in SITES {
        for fold in 0..6 {
            let next = (fold + 1) % 6;
            let p = baselines.get(&key(fold, "11")).unwrap();
            let e = baselines.get(&key(fold, "10")).unwrap();
            let r = baselines.get(&key(fold, "01")).unwrap();
            let n = baselines.get(&key(fold, "00")).unwrap();
            let p_next = baselines.get(&key(next, "11")).unwrap();
            let e_next = baselines.get(&key(next, "10")).unwrap();
            for (arm, donor, recipient) in [
                ("matched_forward", p, e),
                ("matched_inverse", e, p),
                ("positive_self", p, p),
                ("control_self", e, e),
                ("entity_absent_forward", r, n),
                ("entity_absent_inverse", n, r),
                ("nonpaired_forward", p_next, e),
                ("nonpaired_inverse", e_next, p),
            ] {
                arms.push(run_arm(
                    weights,
                    source_index,
                    patched,
                    site,
                    fold,
                    arm,
                    donor,
                    recipient,
                )?);
                forward_count += 1;
            }
            eprintln!("patched L{site} fold {fold}/5 complete");
        }
    }

    if forward_count != EXPECTED_FORWARDS || arms.len() != 192 {
        return Err(format!(
            "forward budget mismatch: total={forward_count}, arms={}",
            arms.len()
        )
        .into());
    }
    let output = RunOutput {
        schema: "larql.causal_entity_relation_patching.v1",
        vindex: args.vindex.display().to_string(),
        suite: args.suite.display().to_string(),
        source_patch: args.source_patch.display().to_string(),
        sites: SITES,
        target_id: TARGET_ID,
        distractor_id: DISTRACTOR_ID,
        baseline_forwards: 24,
        noop_forwards: 24,
        patched_forwards: arms.len(),
        total_forwards: forward_count,
        baselines: baseline_outputs,
        arms,
    };
    let file = std::fs::File::create(&args.output)?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &output)?;
    eprintln!("saved {}", args.output.display());
    Ok(())
}
