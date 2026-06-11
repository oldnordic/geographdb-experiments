//! Corpus-native geometric graph builder — scaled, multi-domain, BPE-based.
//!
//! Builds a sense-expanded Graph4D from one or more corpora:
//!   - Train/load BPE tokenizer
//!   - Sparse co-occurrence counting per domain
//!   - Sparse randomized SVD → 3D positions
//!   - Build sparse PMI graph (top-K neighbors)
//!   - Ego-network sense clustering (connected components)
//!   - Build GraphNode4D with domain labels
//!   - Save to .geo storage + tokenizer.json + vocab.json
//!
//! Usage:
//!   cargo run --release --example corpus_native_graph -- \
//!     --corpus text:/path/to/wiki.txt \
//!     --corpus code:/path/to/code.txt \
//!     --dataset code=nickrosh/Evol-Instruct-Code-80k-v1|instruction,output \
//!     --vocab-size 20000 \
//!     --output graph_dir

use anyhow::{Context, Result};
use geographdb_core::corpus::{
    inject_tool_subgraphs, parse_tool_schemas, HfDatasetLoader, HfDatasetSpec,
};
use geographdb_core::{save_graph4d, GraphNode4D, TemporalEdge};
use glam::Vec3;
use nalgebra::DMatrix;
use petgraph::graph::{NodeIndex, UnGraph};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{AddedToken, Tokenizer, TokenizerBuilder};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const VOCAB_SIZE: usize = 20_000;
const COOC_WINDOW: usize = 5;
const MIN_FREQ: usize = 5;
const SVD_DIM: usize = 3;
const SVD_POWER_ITERATIONS: usize = 2;
const TOP_K_PMI: usize = 20;
const MIN_COMPONENT_SIZE: usize = 4;
const MAX_SENSES_PER_TOKEN: usize = 5;

// ---------------------------------------------------------------------------
// Corpus handling
// ---------------------------------------------------------------------------

fn parse_corpora_args(args: &[String]) -> Vec<(String, String)> {
    let mut corpora = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--corpus" && i + 1 < args.len() {
            let parts: Vec<&str> = args[i + 1].splitn(2, ':').collect();
            if parts.len() == 2 {
                corpora.push((parts[0].to_string(), parts[1].to_string()));
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    corpora
}

fn parse_dataset_args(args: &[String]) -> Vec<HfDatasetSpec> {
    let mut specs = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--dataset" && i + 1 < args.len() {
            match HfDatasetSpec::from_arg(&args[i + 1]) {
                Ok(spec) => specs.push(spec),
                Err(e) => eprintln!("Ignoring invalid dataset arg: {e}"),
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    specs
}

fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse::<T>().ok())
        .unwrap_or(default)
}

fn parse_output_dir(args: &[String]) -> String {
    args.windows(2)
        .find(|w| w[0] == "--output")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "corpus_native_graph".to_string())
}

fn parse_tokenizer_path(args: &[String]) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == "--tokenizer")
        .map(|w| w[1].clone())
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

fn load_or_train_tokenizer(
    corpora: &[(String, Vec<String>)],
    tokenizer_path: Option<&str>,
    vocab_size: usize,
) -> Result<Tokenizer> {
    // Try loading existing tokenizer
    if let Some(path) = tokenizer_path {
        println!("  Loading tokenizer from {path}...");
        return Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e));
    }

    // Check for cached tokenizer.json in output dir (handled by caller)
    let cached = Path::new("tokenizer.json");
    if cached.exists() {
        println!("  Loading cached tokenizer.json...");
        return Tokenizer::from_file(cached).map_err(|e| anyhow::anyhow!(e));
    }

    // Train BPE on all corpora
    println!("  Training BPE tokenizer (vocab_size={vocab_size})...");
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(false)
        .vocab_size(vocab_size)
        .min_frequency(MIN_FREQ as u64)
        .special_tokens(vec![
            AddedToken::from(String::from("<unk>"), true),
            AddedToken::from(String::from("<s>"), true),
            AddedToken::from(String::from("</s>"), true),
        ])
        .build();

    let sequences: Vec<&str> = corpora
        .iter()
        .flat_map(|(_, texts)| texts.iter().map(|s| s.as_str()))
        .collect();

    let mut tokenizer = TokenizerBuilder::<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >::new()
    .with_model(BPE::default())
    .with_pre_tokenizer(Some(ByteLevel::default().into()))
    .with_post_processor(Some(ByteLevel::default().into()))
    .with_decoder(Some(ByteLevel::default().into()))
    .build()
    .map_err(|e| anyhow::anyhow!(e))?;

    tokenizer
        .train(&mut trainer, sequences.iter().copied())
        .map_err(|e| anyhow::anyhow!(e))?;

    Ok(tokenizer.into())
}

fn tokenize(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(text.to_string(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
    Ok(encoding.get_ids().to_vec())
}

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

fn build_vocab(corpora_tokens: &[Vec<u32>], vocab_size: usize) -> (Vec<u32>, HashMap<u32, usize>) {
    let mut freq: HashMap<u32, usize> = HashMap::new();
    for tokens in corpora_tokens {
        for &id in tokens {
            *freq.entry(id).or_insert(0) += 1;
        }
    }

    let mut items: Vec<(u32, usize)> = freq.into_iter().collect();
    items.sort_by(|a, b| b.1.cmp(&a.1));

    let vocab: Vec<u32> = items
        .into_iter()
        .filter(|(_, c)| *c >= MIN_FREQ)
        .take(vocab_size)
        .map(|(id, _)| id)
        .collect();

    let token_to_idx: HashMap<u32, usize> =
        vocab.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    (vocab, token_to_idx)
}

// ---------------------------------------------------------------------------
// Sparse co-occurrence + PMI
// ---------------------------------------------------------------------------

/// Sparse matrix stored as (row, col) -> value map.
#[derive(Debug, Clone)]
struct SparseMatrix {
    rows: usize,
    cols: usize,
    entries: HashMap<(usize, usize), f32>,
}

impl SparseMatrix {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            entries: HashMap::new(),
        }
    }

    fn add(&mut self, row: usize, col: usize, val: f32) {
        *self.entries.entry((row, col)).or_insert(0.0) += val;
    }

    fn get(&self, row: usize, col: usize) -> f32 {
        self.entries.get(&(row, col)).copied().unwrap_or(0.0)
    }

    fn row_sums(&self) -> Vec<f32> {
        let mut sums = vec![0.0f32; self.rows];
        for ((i, _), v) in &self.entries {
            sums[*i] += *v;
        }
        sums
    }

    fn total_sum(&self) -> f32 {
        self.entries.values().sum()
    }

    /// Multiply sparse matrix by dense matrix: A * X where X is (cols x k).
    fn matmul_dense(&self, x: &DMatrix<f32>) -> DMatrix<f32> {
        let k = x.ncols();
        let mut y = DMatrix::zeros(self.rows, k);
        for ((i, j), v) in &self.entries {
            for kk in 0..k {
                y[(*i, kk)] += *v * x[(*j, kk)];
            }
        }
        y
    }

    /// Multiply transpose: A^T * X where X is (rows x k).
    fn t_matmul_dense(&self, x: &DMatrix<f32>) -> DMatrix<f32> {
        let k = x.ncols();
        let mut y = DMatrix::zeros(self.cols, k);
        for ((i, j), v) in &self.entries {
            for kk in 0..k {
                y[(*j, kk)] += *v * x[(*i, kk)];
            }
        }
        y
    }
}

fn build_sparse_cooc(tokens: &[u32], token_to_idx: &HashMap<u32, usize>) -> SparseMatrix {
    let v = token_to_idx.len();
    let mut cooc = SparseMatrix::new(v, v);

    for i in 0..tokens.len() {
        let t1 = tokens[i];
        let Some(&idx1) = token_to_idx.get(&t1) else {
            continue;
        };
        for j in (i + 1)..=(i + COOC_WINDOW).min(tokens.len().saturating_sub(1)) {
            let t2 = tokens[j];
            let Some(&idx2) = token_to_idx.get(&t2) else {
                continue;
            };
            if idx1 == idx2 {
                continue;
            }
            cooc.add(idx1, idx2, 1.0);
            cooc.add(idx2, idx1, 1.0);
        }
    }

    cooc
}

fn build_sparse_pmi(cooc: &SparseMatrix) -> SparseMatrix {
    let v = cooc.rows;
    let row_sums = cooc.row_sums();
    let total = cooc.total_sum();
    let mut pmi = SparseMatrix::new(v, v);

    for ((i, j), count) in &cooc.entries {
        if row_sums[*i] > 0.0 && row_sums[*j] > 0.0 && total > 0.0 {
            let p_cooc = count / total;
            let p_i = row_sums[*i] / total;
            let p_j = row_sums[*j] / total;
            let val = (p_cooc / (p_i * p_j)).max(1e-10).ln().max(0.0);
            if val > 0.0 {
                pmi.add(*i, *j, val);
            }
        }
    }

    pmi
}

// ---------------------------------------------------------------------------
// Sparse randomized SVD
// ---------------------------------------------------------------------------

fn randomized_svd(
    pmi: &SparseMatrix,
    rank: usize,
    power_iter: usize,
    seed: u64,
) -> Vec<(f32, f32, f32)> {
    let n = pmi.rows;
    let mut rng = StdRng::seed_from_u64(seed);

    // Random Gaussian matrix Ω (n x rank)
    let omega: DMatrix<f32> = DMatrix::from_fn(n, rank, |_, _| rng.random::<f32>());

    // Y = A * Ω
    let mut y = pmi.matmul_dense(&omega);

    // Power iterations to sharpen singular values
    for _ in 0..power_iter {
        y = pmi.matmul_dense(&pmi.t_matmul_dense(&y));
    }

    // QR decomposition of Y
    let qr = nalgebra::linalg::QR::new(y);
    let q = qr.q();

    // B = Q^T * A
    let b = q.transpose() * pmi.matmul_dense(&q);

    // SVD of B
    let svd = nalgebra::linalg::SVD::new(b, true, true);
    let u_b = svd.u.expect("SVD failed");
    let s = svd.singular_values;

    // U = Q * U_B
    let u = q * u_b;

    // Extract positions
    let mut pos = vec![(0.0f32, 0.0f32, 0.0f32); n];
    for i in 0..n {
        let mut coords = [0.0f32; 3];
        for k in 0..rank.min(s.len()) {
            coords[k] = (u[(i, k)] * s[k]) as f32;
        }
        let norm = (coords[0] * coords[0] + coords[1] * coords[1] + coords[2] * coords[2])
            .sqrt()
            .max(1e-9);
        pos[i] = (coords[0] / norm, coords[1] / norm, coords[2] / norm);
    }

    pos
}

// ---------------------------------------------------------------------------
// Sparse PMI graph
// ---------------------------------------------------------------------------

fn build_sparse_graph(vocab: &[u32], pmi: &SparseMatrix, top_k: usize) -> UnGraph<usize, f32> {
    let v = vocab.len();
    let mut graph = UnGraph::<usize, f32>::new_undirected();
    let node_indices: Vec<_> = (0..v).map(|i| graph.add_node(i)).collect();

    for i in 0..v {
        let mut neighbors: Vec<(usize, f32)> = Vec::new();
        for j in 0..v {
            if i == j {
                continue;
            }
            let w = pmi.get(i, j);
            if w > 0.0 {
                neighbors.push((j, w));
            }
        }
        neighbors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (j, w) in neighbors.into_iter().take(top_k) {
            graph.add_edge(node_indices[i], node_indices[j], w);
        }
    }

    graph
}

// ---------------------------------------------------------------------------
// Sense clustering
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SenseCluster {
    sense_id: usize,
    neighbors: Vec<usize>,
}

fn detect_senses(
    vocab: &[u32],
    pmi: &SparseMatrix,
    _positions: &[(f32, f32, f32)],
    graph: &UnGraph<usize, f32>,
    top_k: usize,
) -> HashMap<u32, Vec<SenseCluster>> {
    let v = vocab.len();
    let mut token_to_senses: HashMap<u32, Vec<SenseCluster>> = HashMap::new();

    for i in 0..v {
        let tok = vocab[i];

        let mut neighbors: Vec<usize> = Vec::new();
        for j in 0..v {
            if i == j {
                continue;
            }
            if pmi.get(i, j) > 0.0 {
                neighbors.push(j);
            }
        }

        if neighbors.len() < 2 {
            token_to_senses.insert(
                tok,
                vec![SenseCluster {
                    sense_id: 0,
                    neighbors: Vec::new(),
                }],
            );
            continue;
        }

        if neighbors.len() > top_k {
            let mut scored: Vec<_> = neighbors.iter().map(|&j| (j, pmi.get(i, j))).collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            neighbors = scored.into_iter().take(top_k).map(|(j, _)| j).collect();
        }

        let mut ego = UnGraph::<usize, f32>::new_undirected();
        let mut ego_nodes: HashMap<usize, NodeIndex> = HashMap::new();
        for &n in &neighbors {
            ego_nodes.insert(n, ego.add_node(n));
        }

        for a in 0..neighbors.len() {
            for b in (a + 1)..neighbors.len() {
                let na = neighbors[a];
                let nb = neighbors[b];
                if graph.contains_edge(NodeIndex::new(na), NodeIndex::new(nb)) {
                    ego.add_edge(ego_nodes[&na], ego_nodes[&nb], 1.0);
                }
            }
        }

        let sccs = petgraph::algo::tarjan_scc(&ego);
        let mut components: Vec<HashSet<usize>> = sccs
            .into_iter()
            .map(|c| c.into_iter().map(|nx| ego[nx]).collect())
            .collect();

        if components.len() == 1 {
            token_to_senses.insert(
                tok,
                vec![SenseCluster {
                    sense_id: 0,
                    neighbors,
                }],
            );
            continue;
        }

        components.sort_by_key(|c| std::cmp::Reverse(c.len()));

        let mut kept: Vec<HashSet<usize>> = Vec::new();
        let mut residual = HashSet::new();
        for comp in components {
            if comp.len() >= MIN_COMPONENT_SIZE && kept.len() < MAX_SENSES_PER_TOKEN - 1 {
                kept.push(comp);
            } else {
                residual.extend(comp);
            }
        }
        if !residual.is_empty() {
            kept.push(residual);
        }

        let clusters: Vec<SenseCluster> = kept
            .into_iter()
            .enumerate()
            .map(|(sid, comp)| SenseCluster {
                sense_id: sid,
                neighbors: comp.into_iter().collect(),
            })
            .collect();

        token_to_senses.insert(tok, clusters);
    }

    token_to_senses
}

// ---------------------------------------------------------------------------
// Build GraphNode4D
// ---------------------------------------------------------------------------

fn build_graph4d(
    vocab: &[u32],
    token_to_senses: &HashMap<u32, Vec<SenseCluster>>,
    positions: &[(f32, f32, f32)],
    pmi: &SparseMatrix,
    domain_map: &HashMap<u32, String>,
) -> Vec<GraphNode4D> {
    let mut nodes: Vec<GraphNode4D> = Vec::new();
    let mut nid_to_idx: HashMap<u64, usize> = HashMap::new();

    for i in 0..vocab.len() {
        let tok = vocab[i];
        let (bx, by, bz) = positions[i];
        let clusters = token_to_senses.get(&tok).cloned().unwrap_or_else(|| {
            vec![SenseCluster {
                sense_id: 0,
                neighbors: Vec::new(),
            }]
        });

        let domain = domain_map
            .get(&tok)
            .cloned()
            .unwrap_or_else(|| "mixed".to_string());

        for cluster in &clusters {
            let sid = cluster.sense_id;
            let (cx, cy, cz) = if cluster.neighbors.is_empty() {
                (bx, by, bz)
            } else {
                let mut sx = 0.0f32;
                let mut sy = 0.0f32;
                let mut sz = 0.0f32;
                for &nj in &cluster.neighbors {
                    let (px, py, pz) = positions[nj];
                    sx += px;
                    sy += py;
                    sz += pz;
                }
                let n = cluster.neighbors.len() as f32;
                (sx / n, sy / n, sz / n)
            };

            let alpha = 0.35f32;
            let x = bx * (1.0 - alpha) + cx * alpha;
            let y = by * (1.0 - alpha) + cy * alpha;
            let z = bz * (1.0 - alpha) + cz * alpha;
            let nid = (tok as u64) * 1000 + (sid as u64);

            let mut properties = geographdb_core::GraphProperties::new();
            properties.insert(
                "domain".to_string(),
                serde_json::Value::String(domain.clone()),
            );

            nid_to_idx.insert(nid, nodes.len());
            nodes.push(GraphNode4D {
                id: nid,
                x,
                y,
                z,
                begin_ts: 0,
                end_ts: u64::MAX,
                properties,
                successors: Vec::new(),
            });
        }
    }

    for i in 0..vocab.len() {
        let tok_i = vocab[i];
        let clusters_i = token_to_senses.get(&tok_i);
        let num_senses_i = clusters_i.map(|c| c.len()).unwrap_or(1);
        for j in 0..vocab.len() {
            if i == j {
                continue;
            }
            let w = pmi.get(i, j);
            if w <= 0.0 {
                continue;
            }
            let tok_j = vocab[j];
            let clusters_j = token_to_senses.get(&tok_j);
            let num_senses_j = clusters_j.map(|c| c.len()).unwrap_or(1);

            for sid_i in 0..num_senses_i {
                let nid_i = (tok_i as u64) * 1000 + (sid_i as u64);
                let idx_i = nid_to_idx[&nid_i];
                for sid_j in 0..num_senses_j {
                    let nid_j = (tok_j as u64) * 1000 + (sid_j as u64);
                    nodes[idx_i].successors.push(TemporalEdge {
                        dst: nid_j,
                        weight: w,
                        begin_ts: 0,
                        end_ts: u64::MAX,
                    });
                }
            }
        }
    }

    nodes
}

// ---------------------------------------------------------------------------
// Tool subgraph helpers
// ---------------------------------------------------------------------------

fn compute_tool_domain_centroid(
    nodes: &[GraphNode4D],
    domain_map: &std::collections::HashMap<u32, String>,
) -> Option<Vec3> {
    let mut sum = Vec3::ZERO;
    let mut count = 0usize;
    for node in nodes {
        let token_id = (node.id / 1000) as u32;
        if domain_map.get(&token_id) == Some(&"tool".to_string()) {
            sum += node.position();
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f32)
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let corpora_specs = parse_corpora_args(&args);
    let dataset_specs = parse_dataset_args(&args);
    if corpora_specs.is_empty() && dataset_specs.is_empty() {
        eprintln!("Usage: corpus_native_graph --corpus name:path [--corpus name:path ...] --dataset domain=repo[:subset][|cols] [--dataset ...] --output dir [--vocab-size N] [--tokenizer path]");
        std::process::exit(1);
    }

    let vocab_size = parse_flag(&args, "--vocab-size", VOCAB_SIZE);
    let out_dir = parse_output_dir(&args);
    let tokenizer_path = parse_tokenizer_path(&args);

    println!("Corpus-Native Graph Builder (Scaled, Multi-Domain)");
    println!("==================================================\n");

    // 1. Load corpora
    println!("[1/8] Loading corpora...");
    let mut corpora: Vec<(String, Vec<String>)> = Vec::new();
    for (name, path) in &corpora_specs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read corpus {name} from {path}"))?;
        println!("  {}: {} chars", name, text.len());
        corpora.push((name.clone(), vec![text]));
    }

    for spec in &dataset_specs {
        let loader = HfDatasetLoader::new(spec.clone())
            .with_context(|| format!("Failed to create HF loader for {}", spec.repo_id))?;
        let mut texts: Vec<String> = Vec::new();
        let mut total_chars = 0usize;
        for text in loader.stream_texts()? {
            let text =
                text.with_context(|| format!("Failed to read text from {}", spec.repo_id))?;
            total_chars += text.len();
            texts.push(text);
        }
        println!(
            "  {} (HF {}): {} texts, {} chars",
            spec.domain,
            spec.repo_id,
            texts.len(),
            total_chars
        );
        corpora.push((spec.domain.clone(), texts));
    }

    // 2. Train/load tokenizer
    println!("[2/8] Preparing tokenizer...");
    let tokenizer = load_or_train_tokenizer(&corpora, tokenizer_path.as_deref(), vocab_size)?;
    let out_path = Path::new(&out_dir);
    std::fs::create_dir_all(out_path)?;
    tokenizer
        .save(out_path.join("tokenizer.json"), false)
        .map_err(|e| anyhow::anyhow!(e))?;

    // 3. Tokenize corpora
    println!("[3/8] Tokenizing corpora...");
    let mut all_tokens: Vec<Vec<u32>> = Vec::new();
    let mut domain_token_counts: HashMap<String, HashMap<u32, usize>> = HashMap::new();

    for (name, texts) in &corpora {
        let mut tokens = Vec::new();
        for text in texts {
            tokens.extend(tokenize(&tokenizer, text)?);
        }
        println!("  {}: {} tokens", name, tokens.len());
        all_tokens.push(tokens.clone());

        let mut counts: HashMap<u32, usize> = HashMap::new();
        for &id in &tokens {
            *counts.entry(id).or_insert(0) += 1;
        }
        domain_token_counts.insert(name.clone(), counts);
    }

    // 4. Build vocab
    println!("[4/8] Building vocabulary (top {vocab_size})...");
    let (vocab, token_to_idx) = build_vocab(&all_tokens, vocab_size);
    println!("  Vocab size: {}", vocab.len());

    // Determine primary domain per token
    let mut domain_map: HashMap<u32, String> = HashMap::new();
    for &tok in &vocab {
        let mut best_domain = "mixed".to_string();
        let mut best_count = 0usize;
        for (domain, counts) in &domain_token_counts {
            if let Some(&count) = counts.get(&tok) {
                if count > best_count {
                    best_count = count;
                    best_domain = domain.clone();
                }
            }
        }
        domain_map.insert(tok, best_domain);
    }

    // 5. Sparse co-occurrence + PMI
    println!("[5/8] Computing sparse PMI...");
    let mut combined_cooc = SparseMatrix::new(vocab.len(), vocab.len());
    for tokens in &all_tokens {
        let cooc = build_sparse_cooc(tokens, &token_to_idx);
        for ((i, j), v) in &cooc.entries {
            combined_cooc.add(*i, *j, *v);
        }
    }
    let pmi = build_sparse_pmi(&combined_cooc);
    println!("  PMI non-zero: {}", pmi.entries.len());

    // 6. Sparse randomized SVD
    println!("[6/8] Randomized SVD to {SVD_DIM}D...");
    let positions = randomized_svd(&pmi, SVD_DIM, SVD_POWER_ITERATIONS, 42);

    // 7. Sparse graph + sense clustering
    println!("[7/8] Building sparse PMI graph and senses...");
    let graph = build_sparse_graph(&vocab, &pmi, TOP_K_PMI);
    println!(
        "  Nodes: {}  Edges: {}  Mean degree: {:.1}",
        graph.node_count(),
        graph.edge_count(),
        2.0 * graph.edge_count() as f32 / graph.node_count() as f32
    );

    let token_to_senses = detect_senses(&vocab, &pmi, &positions, &graph, TOP_K_PMI);
    let multi_sense = token_to_senses.values().filter(|s| s.len() > 1).count();
    let total_senses: usize = token_to_senses.values().map(|s| s.len()).sum();
    println!(
        "  Multi-sense tokens: {multi_sense}/{}  Total sense-nodes: {total_senses}",
        vocab.len()
    );

    // 8. Build GraphNode4D
    println!("[8/8] Building GraphNode4D...");
    let mut nodes = build_graph4d(&vocab, &token_to_senses, &positions, &pmi, &domain_map);
    println!("  Nodes: {}", nodes.len());

    // 8b. Inject tool schema subgraphs when tool data is present
    let tool_system_texts: Vec<String> = corpora
        .iter()
        .filter(|(domain, _)| domain == "tool")
        .flat_map(|(_, texts)| texts.iter().cloned())
        .collect();
    if !tool_system_texts.is_empty() {
        println!("[8b] Injecting tool schema subgraphs...");
        let tool_schemas = parse_tool_schemas(&tool_system_texts);
        println!("  Parsed {} unique tool schemas", tool_schemas.len());

        // Centroid of existing tool-domain nodes for fallback placement.
        let tool_centroid = compute_tool_domain_centroid(&nodes, &domain_map);
        inject_tool_subgraphs(&mut nodes, &tool_schemas, &tokenizer, tool_centroid)
            .context("Failed to inject tool subgraphs")?;
        println!("  Nodes after injection: {}", nodes.len());
    }

    // 9. Save
    println!("\nSaving graph to {out_dir}...");
    save_graph4d(&nodes, out_path).with_context(|| format!("Failed to save graph to {out_dir}"))?;

    // Save vocab mapping
    let vocab_json: serde_json::Map<String, serde_json::Value> = vocab
        .iter()
        .map(|tok| {
            let word = tokenizer.decode(&[*tok], false).unwrap_or_default();
            (tok.to_string(), serde_json::Value::String(word))
        })
        .collect();
    std::fs::write(
        out_path.join("vocab.json"),
        serde_json::to_string_pretty(&vocab_json)?,
    )?;

    // Save domain mapping
    let domain_json: serde_json::Map<String, serde_json::Value> = domain_map
        .iter()
        .map(|(tok, domain)| (tok.to_string(), serde_json::Value::String(domain.clone())))
        .collect();
    std::fs::write(
        out_path.join("domains.json"),
        serde_json::to_string_pretty(&domain_json)?,
    )?;

    println!("Saved: tokenizer.json, vocab.json, domains.json, .geo storage");

    Ok(())
}
