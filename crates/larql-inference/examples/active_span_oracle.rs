//! E2 oracle: exercise an exact continuation span through the existing Q4K
//! constrained decoder, with no patch-trigger logic.
//!
//! The caller supplies a prompt and a multi-token target.  The first target
//! token is treated as the already-observed trigger token.  `ActiveSpan`
//! hard-masks each remaining target ID in order, then releases the mask for
//! one trailing token so cleanup/loop behaviour is observable.

use std::cell::Cell;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::Serialize;

type AnyError = Box<dyn Error>;

#[derive(Debug)]
struct ActiveSpan {
    continuation: Vec<u32>,
}

impl ActiveSpan {
    fn apply(&self, generated: &[u32], logits: &mut [f32]) {
        let Some(&expected) = self.continuation.get(generated.len()) else {
            return;
        };
        for (id, logit) in logits.iter_mut().enumerate() {
            if id != expected as usize {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}

#[derive(Serialize)]
struct OracleResult {
    prompt: String,
    target: String,
    target_ids: Vec<u32>,
    armed_after_id: u32,
    continuation_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    generated_text: String,
    emitted_span_ids: Vec<u32>,
    emitted_span_text: String,
    exact_emitted_span: bool,
    trailing_id: Option<u32>,
    looped_to_first_target_token: bool,
    constrained_ms: f64,
    mask_ms: f64,
    mask_calls: usize,
    mask_fraction: f64,
}

fn load_q4k(
    vindex: &Path,
) -> Result<
    (
        larql_inference::ModelWeights,
        larql_inference::tokenizers::Tokenizer,
        larql_vindex::VectorIndex,
    ),
    AnyError,
> {
    let config = larql_vindex::load_vindex_config(vindex)?;
    if config.quant != larql_vindex::QuantFormat::Q4K {
        return Err("active_span_oracle requires a Q4K vindex".into());
    }

    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let mut index = larql_vindex::VectorIndex::load_vindex(vindex, &mut callbacks)?;
    index.load_attn_kquant(vindex)?;
    index.load_interleaved_kquant(vindex)?;
    let _ = index.load_lm_head_kquant(vindex);
    let weights = larql_vindex::load_model_weights_kquant(vindex, &mut callbacks)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex)?;
    Ok((weights, tokenizer, index))
}

fn encode_target(
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    target: &str,
) -> Result<Vec<u32>, AnyError> {
    let encoding = tokenizer
        .encode(format!(" {target}"), false)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    Ok(encoding.get_ids().to_vec())
}

fn run_case(
    weights: &mut larql_inference::ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    index: &larql_vindex::VectorIndex,
    prompt: &str,
    target: &str,
) -> Result<OracleResult, AnyError> {
    let target_ids = encode_target(&tokenizer, target)?;
    if target_ids.len() < 2 {
        return Err(
            format!("target must contain at least two tokenizer IDs, got {target_ids:?}").into(),
        );
    }

    let mut prompt_ids = larql_inference::encode_prompt(tokenizer, weights.arch.as_ref(), prompt)?;
    prompt_ids.push(target_ids[0]);

    let active = ActiveSpan {
        continuation: target_ids[1..].to_vec(),
    };
    let max_tokens = active.continuation.len() + 1;
    let mask_time = Cell::new(Duration::ZERO);
    let mask_calls = Cell::new(0usize);
    let generation_start = Instant::now();
    let generated = larql_inference::vindex::generate_kquant_cpu_constrained_cached(
        weights,
        tokenizer,
        &prompt_ids,
        max_tokens,
        index,
        |generated, logits| {
            let start = Instant::now();
            active.apply(generated, logits);
            mask_time.set(mask_time.get() + start.elapsed());
            mask_calls
                .set(mask_calls.get() + usize::from(generated.len() < active.continuation.len()));
        },
    );
    let constrained_ms = generation_start.elapsed().as_secs_f64() * 1000.0;
    let mask_ms = mask_time.get().as_secs_f64() * 1000.0;

    let generated_ids: Vec<u32> = generated.iter().map(|(_, id)| *id).collect();
    let mut emitted_span_ids = vec![target_ids[0]];
    if let Some(continuation) = generated_ids.get(..active.continuation.len()) {
        emitted_span_ids.extend_from_slice(continuation);
    }
    let emitted = emitted_span_ids == target_ids;
    let trailing_id = generated_ids.get(active.continuation.len()).copied();
    let generated_text = tokenizer
        .decode(&generated_ids, true)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let emitted_span_text = tokenizer
        .decode(&emitted_span_ids, true)
        .map_err(|error| std::io::Error::other(error.to_string()))?;

    Ok(OracleResult {
        prompt: prompt.to_string(),
        target: target.to_string(),
        target_ids: target_ids.clone(),
        armed_after_id: target_ids[0],
        continuation_ids: active.continuation,
        generated_ids,
        generated_text,
        emitted_span_ids,
        emitted_span_text,
        exact_emitted_span: emitted,
        trailing_id,
        looped_to_first_target_token: trailing_id == target_ids.first().copied(),
        constrained_ms,
        mask_ms,
        mask_calls: mask_calls.get(),
        mask_fraction: if constrained_ms > 0.0 {
            mask_ms / constrained_ms
        } else {
            0.0
        },
    })
}

fn main() -> Result<(), AnyError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 || !(args.len() - 2).is_multiple_of(2) {
        return Err(
            "usage: active_span_oracle <q4k-vindex> <prompt> <target> [<prompt> <target> ...]"
                .into(),
        );
    }
    let vindex = PathBuf::from(&args[1]);
    let (mut weights, tokenizer, index) = load_q4k(&vindex)?;
    let mut results = Vec::with_capacity((args.len() - 2) / 2);
    for pair in args[2..].chunks_exact(2) {
        results.push(run_case(
            &mut weights,
            &tokenizer,
            &index,
            &pair[0],
            &pair[1],
        )?);
    }
    if results.len() == 1 {
        println!("{}", serde_json::to_string_pretty(&results[0])?);
    } else {
        println!("{}", serde_json::to_string_pretty(&results)?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_span_masks_exactly_the_next_id() {
        let span = ActiveSpan {
            continuation: vec![2, 4],
        };
        let mut logits = vec![1.0; 6];

        span.apply(&[2], &mut logits);

        assert_eq!(logits[4], 1.0);
        assert!(logits
            .iter()
            .enumerate()
            .all(|(id, value)| id == 4 || value.is_infinite() && value.is_sign_negative()));
    }

    #[test]
    fn active_span_releases_mask_after_completion() {
        let span = ActiveSpan {
            continuation: vec![2, 4],
        };
        let mut logits = vec![0.0, 1.0, 2.0];
        let original = logits.clone();

        span.apply(&[2, 4], &mut logits);

        assert_eq!(logits, original);
    }
}
