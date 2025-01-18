use anyhow::Result;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

use memmap2::Mmap;
use ndarray_stats::errors::MinMaxError;

struct Config {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: u32,
    seq_len: u32,
}

struct TransformerWeights {
    // token embedding table
    token_embedding_table: Array2<f32>,
    // weights for rmsnorms
    rms_att_weight: Array2<f32>,
    rms_ffn_weight: Array2<f32>,
    // weights for matmuls
    wq: Vec<Array2<f32>>,
    wk: Vec<Array2<f32>>,
    wv: Vec<Array2<f32>>,
    wo: Vec<Array2<f32>>,
    // weights for ffn
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
    // final rmsnorm
    rms_final_weight: Array1<f32>,
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Array2<f32>,
    freq_cis_imag: Array2<f32>,
    // (optional) classifier weights for the logits, on the last layer
    wcls: Array2<f32>,
}

struct ProbIndex {
    prob: f32,
    index: u32,
}

struct RunState {
    x: Array1<f32>,
    xb: Array1<f32>,
    xb2: Array1<f32>,
    hb: Array1<f32>,
    hb2: Array1<f32>,
    q: Array1<f32>,
    k: Array1<f32>,
    v: Array1<f32>,
    att: Array1<f32>,
    logits: Array1<f32>,
    probindex: Vec<ProbIndex>,
    key_cache: Vec<Array2<f32>>,
    value_cache: Vec<Array2<f32>>,
}

fn rmsnorm(x: &Array1<f32>, weight: &Array1<f32>) -> Array1<f32> {
    let ss = (x * x).sum() / (x.len() as f32) + 1e-5;
    let ss_inv_sqrt = 1.0 / ss.sqrt();

    weight * x * ss_inv_sqrt
}

fn softmax(x: Array1<f32>) -> Result<Array1<f32>> {
    let max_val = x.max()?;
    let x_exp = x.map(|v| (v - max_val).exp());
    let x_sum = x_exp.sum();
    Ok(x_exp.map(|v| v / &x_sum))
}

// fn matmul(x: Vec<f32>, y: Vec<f32>) -> Vec<f32> {

// }

fn transformer(token: i32, pos: i32, p: Config, s: &mut RunState, w: &TransformerWeights) {
    s.x = rmsnorm(&s.x, &w.rms_final_weight);
    s.logits = s.x.dot(&w.wcls);
}