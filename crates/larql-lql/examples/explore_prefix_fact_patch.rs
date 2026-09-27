//! Bounded L21 prefix-row transplantation diagnostic on aligned source contexts.
use larql_inference::{
    forward::LayerHook,
    hidden_to_raw_logits,
    vindex::{predict_kquant_hidden_with_ffn_hooked, WalkFfn},
    InferenceWeights,
};
use ndarray::{Array2, Axis};
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::HashSet,
    error::Error,
    io::{BufRead, Write},
    path::Path,
};
type E = Box<dyn Error>;
#[derive(Clone, Deserialize)]
struct RowDelta {
    position: usize,
    delta: Vec<f32>,
}
#[derive(Deserialize)]
struct Request {
    ids: Vec<u32>,
    #[serde(default)]
    row_deltas: Vec<RowDelta>,
    #[serde(default)]
    gain: f32,
}
fn valid_rows(rows: &[RowDelta], length: usize, width: usize) -> bool {
    let mut seen = HashSet::new();
    rows.iter().all(|r| {
        r.position < length
            && seen.insert(r.position)
            && r.delta.len() == width
            && r.delta.iter().all(|x| x.is_finite())
    })
}
struct Hook {
    rows: Vec<RowDelta>,
    gain: f32,
    captures: Vec<(usize, Vec<f32>)>,
    prefix: Vec<Vec<f32>>,
    applied: bool,
}
impl LayerHook for Hook {
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {
        if layer == 21 {
            for r in &self.rows {
                for (x, d) in h.row_mut(r.position).iter_mut().zip(&r.delta) {
                    *x += self.gain * d;
                }
            }
            self.applied = !self.rows.is_empty();
            self.prefix = h.rows().into_iter().map(|r| r.to_vec()).collect();
        }
        if [21, 31, 41].contains(&layer) {
            self.captures.push((layer, h.row(h.nrows() - 1).to_vec()));
        }
    }
}
fn emit(v: serde_json::Value) -> Result<(), E> {
    let mut o = std::io::stdout().lock();
    write!(o, "EXP\t")?;
    serde_json::to_writer(&mut o, &v)?;
    writeln!(o)?;
    o.flush()?;
    Ok(())
}
fn main() -> Result<(), E> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        return Err("usage: explore_prefix_fact_patch VINDEX".into());
    }
    let path = Path::new(&args[0]);
    let cfg = larql_vindex::load_vindex_config(path)?;
    if cfg.hidden_size != 2560 || cfg.num_layers != 42 || cfg.vocab_size != 262144 {
        return Err("unexpected model".into());
    }
    let mut w = InferenceWeights::load(path, &cfg, &mut larql_vindex::SilentLoadCallbacks)?;
    let (weights, index) = w.as_quantised_parts_mut().ok_or("Q4K required")?;
    emit(json!({"type":"ready","max_forwards":24}))?;
    for (ordinal, line) in std::io::stdin().lock().lines().enumerate() {
        if ordinal >= 24 {
            return Err("session forward budget exhausted".into());
        }
        let r: Request = serde_json::from_str(&line?)?;
        if r.ids.is_empty()
            || r.ids.len() > 256
            || r.ids.iter().any(|v| *v >= 262144)
            || !r.gain.is_finite()
            || !valid_rows(&r.row_deltas, r.ids.len(), 2560)
        {
            return Err("invalid prefix patch request".into());
        }
        let positions: Vec<_> = r.row_deltas.iter().map(|r| r.position).collect();
        let mut hook = Hook {
            rows: r.row_deltas,
            gain: r.gain,
            captures: vec![],
            prefix: vec![],
            applied: false,
        };
        let wr: &larql_models::ModelWeights = unsafe { &*(weights as *const _) };
        let ffn = WalkFfn::new_unlimited(wr, index);
        let h = predict_kquant_hidden_with_ffn_hooked(weights, &r.ids, index, &ffn, &mut hook)?;
        let last = h.row(h.nrows() - 1).to_owned().insert_axis(Axis(0));
        let logits = hidden_to_raw_logits(weights, &last);
        if logits.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite logits".into());
        }
        emit(
            json!({"type":"forward","ordinal":ordinal,"logits":logits,"captures":hook.captures,"prefix_captures":[[21,hook.prefix]],"applied":hook.applied,"applied_position":null,"applied_positions":positions,"detector_score":null,"triggered":false}),
        )?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn row_validation_rejects_ambiguous_or_invalid_patches() {
        let row = RowDelta {
            position: 1,
            delta: vec![1., 2.],
        };
        assert!(valid_rows(std::slice::from_ref(&row), 3, 2));
        assert!(!valid_rows(&[row.clone(), row.clone()], 3, 2));
        assert!(!valid_rows(std::slice::from_ref(&row), 1, 2));
        assert!(!valid_rows(std::slice::from_ref(&row), 3, 3));
        assert!(!valid_rows(
            &[RowDelta {
                position: 0,
                delta: vec![f32::NAN, 0.]
            }],
            3,
            2
        ));
    }
    #[test]
    fn prefix_patch_changes_only_selected_rows_and_keeps_positions() {
        let mut hook = Hook {
            rows: vec![RowDelta {
                position: 1,
                delta: vec![2., -1.],
            }],
            gain: 0.5,
            captures: vec![],
            prefix: vec![],
            applied: false,
        };
        let mut h = Array2::zeros((5, 2));
        hook.on_post_layer(20, &mut h);
        assert_eq!(h.sum(), 0.);
        hook.on_post_layer(21, &mut h);
        assert_eq!(h.row(1).to_vec(), vec![1., -0.5]);
        assert_eq!(h.row(4).to_vec(), vec![0., 0.]);
        assert_eq!(hook.prefix[1], vec![1., -0.5]);
        assert!(hook.applied);
    }
}
