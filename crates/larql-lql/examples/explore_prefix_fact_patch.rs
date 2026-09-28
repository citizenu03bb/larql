//! Bounded L21/L22 prefix-row transplantation diagnostic on aligned contexts.
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
struct KvReplacement {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}
#[derive(Deserialize)]
struct Boundary {
    residual: Option<Vec<Vec<f32>>>,
    kv: Option<KvReplacement>,
}
fn valid_matrix(x: &[Vec<f32>], n: usize, width: usize) -> bool {
    x.len() == n
        && x.iter()
            .all(|r| r.len() == width && r.iter().all(|v| v.is_finite()))
}
fn matrix(x: &[Vec<f32>]) -> Array2<f32> {
    Array2::from_shape_vec((x.len(), x[0].len()), x.iter().flatten().copied().collect()).unwrap()
}
#[derive(Deserialize)]
struct Request {
    ids: Vec<u32>,
    #[serde(default)]
    boundary: Option<Boundary>,
    #[serde(default)]
    layer: Option<usize>,
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
    boundary: Option<Boundary>,
    kv_capture: Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>,
    residual_replaced: bool,
    kv_replaced: bool,
    patch_layer: usize,
    rows: Vec<RowDelta>,
    gain: f32,
    captures: Vec<(usize, Vec<f32>)>,
    prefix: Vec<(usize, Vec<Vec<f32>>)>,
    applied: bool,
}
impl LayerHook for Hook {
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {
        if layer == self.patch_layer {
            for r in &self.rows {
                for (x, d) in h.row_mut(r.position).iter_mut().zip(&r.delta) {
                    *x += self.gain * d;
                }
            }
            self.applied = !self.rows.is_empty();
        }
        if layer == 22 {
            if let Some(x) = self.boundary.as_ref().and_then(|b| b.residual.as_ref()) {
                h.assign(&matrix(x));
                self.residual_replaced = true;
                self.applied = true;
            }
        }
        if [21, 22].contains(&layer) {
            self.prefix
                .push((layer, h.rows().into_iter().map(|r| r.to_vec()).collect()));
        }
        if [21, 22, 31, 41].contains(&layer) {
            self.captures.push((layer, h.row(h.nrows() - 1).to_vec()));
        }
    }
    fn on_kv_cache_store(&mut self, layer: usize, kv: &mut larql_inference::attention::SharedKV) {
        if layer == 22 {
            if let Some(x) = self.boundary.as_ref().and_then(|b| b.kv.as_ref()) {
                assert_eq!(kv.0.dim(), (x.keys.len(), x.keys[0].len()));
                assert_eq!(kv.1.dim(), (x.values.len(), x.values[0].len()));
                kv.0.assign(&matrix(&x.keys));
                kv.1.assign(&matrix(&x.values));
                self.kv_replaced = true;
                self.applied = true;
            }
            self.kv_capture = Some((
                kv.0.rows().into_iter().map(|x| x.to_vec()).collect(),
                kv.1.rows().into_iter().map(|x| x.to_vec()).collect(),
            ));
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
            || !matches!(r.layer, None | Some(21 | 22))
            || !r.gain.is_finite()
            || !valid_rows(&r.row_deltas, r.ids.len(), 2560)
        {
            return Err("invalid prefix patch request".into());
        }
        if let Some(b) = &r.boundary {
            if !r.row_deltas.is_empty()
                || r.layer != Some(22)
                || (b.residual.is_none() && b.kv.is_none())
                || b.residual
                    .as_ref()
                    .is_some_and(|x| !valid_matrix(x, r.ids.len(), 2560))
                || b.kv.as_ref().is_some_and(|x| {
                    !valid_matrix(&x.keys, r.ids.len(), 512)
                        || !valid_matrix(&x.values, r.ids.len(), 512)
                })
            {
                return Err("invalid L22 boundary replacement".into());
            }
        }
        let positions: Vec<_> = r.row_deltas.iter().map(|r| r.position).collect();
        let mut hook = Hook {
            boundary: r.boundary,
            kv_capture: None,
            residual_replaced: false,
            kv_replaced: false,
            patch_layer: r.layer.unwrap_or(21),
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
            json!({"type":"forward","ordinal":ordinal,"logits":logits,"captures":hook.captures,"prefix_captures":hook.prefix,"kv22":hook.kv_capture,"residual_replaced":hook.residual_replaced,"kv_replaced":hook.kv_replaced,"applied":hook.applied,"applied_position":null,"applied_positions":positions,"detector_score":null,"triggered":false}),
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
            boundary: None,
            kv_capture: None,
            residual_replaced: false,
            kv_replaced: false,
            patch_layer: 21,
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
        assert_eq!(hook.prefix[0].1[1], vec![1., -0.5]);
        assert!(hook.applied);
    }
}

#[cfg(test)]
mod downstream_tests {
    use super::*;
    #[test]
    fn l22_patch_leaves_l21_untouched_and_captures_both_layers() {
        let mut hook = Hook {
            boundary: None,
            kv_capture: None,
            residual_replaced: false,
            kv_replaced: false,
            patch_layer: 22,
            rows: vec![RowDelta {
                position: 1,
                delta: vec![2., -1.],
            }],
            gain: 0.5,
            captures: vec![],
            prefix: vec![],
            applied: false,
        };
        let mut h = Array2::zeros((3, 2));
        hook.on_post_layer(21, &mut h);
        assert!(!hook.applied);
        assert_eq!(h.sum(), 0.);
        hook.on_post_layer(22, &mut h);
        assert!(hook.applied);
        assert_eq!(h.row(1).to_vec(), vec![1., -0.5]);
        assert_eq!(h.row(0).to_vec(), vec![0., 0.]);
        assert_eq!(h.row(2).to_vec(), vec![0., 0.]);
        assert_eq!(hook.prefix[0], (21, vec![vec![0., 0.]; 3]));
        assert_eq!(hook.prefix[1].0, 22);
        assert_eq!(hook.prefix[1].1[1], vec![1., -0.5]);
        hook.on_post_layer(23, &mut h);
        assert_eq!(h.row(1).to_vec(), vec![1., -0.5]);
        assert_eq!(hook.prefix.len(), 2);
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    #[test]
    fn rejects_ragged_and_nonfinite_boundary_matrices() {
        assert!(valid_matrix(&[vec![1., 2.]], 1, 2));
        assert!(!valid_matrix(&[vec![1.]], 1, 2));
        assert!(!valid_matrix(&[vec![f32::NAN, 2.]], 1, 2));
    }
    #[test]
    fn boundary_replacement_is_exact_and_separates_residual_from_cache() {
        let mut hook = Hook {
            boundary: Some(Boundary {
                residual: Some(vec![vec![3., 4.]]),
                kv: Some(KvReplacement {
                    keys: vec![vec![5., 6.]],
                    values: vec![vec![7., 8.]],
                }),
            }),
            kv_capture: None,
            residual_replaced: false,
            kv_replaced: false,
            patch_layer: 22,
            rows: vec![],
            gain: 1.,
            captures: vec![],
            prefix: vec![],
            applied: false,
        };
        let mut h = Array2::zeros((1, 2));
        let mut kv = (h.clone(), h.clone());
        hook.on_post_layer(22, &mut h);
        assert_eq!(h.row(0).to_vec(), vec![3., 4.]);
        assert_eq!(kv.0.sum(), 0.);
        assert!(!hook.kv_replaced);
        hook.on_kv_cache_store(22, &mut kv);
        assert_eq!(kv.0.row(0).to_vec(), vec![5., 6.]);
        assert_eq!(kv.1.row(0).to_vec(), vec![7., 8.]);
        assert_eq!(h.row(0).to_vec(), vec![3., 4.]);
        assert!(hook.residual_replaced && hook.kv_replaced);
    }
}
