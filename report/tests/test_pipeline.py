"""
ppp_subtypes/tests/test_pipeline.py
Unit tests for all PPP Subtype Pipeline modules.
Run: python -m unittest ppp_subtypes/tests/test_pipeline.py -v
"""
from __future__ import annotations
import json, sys, tempfile, time, unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ppp_subtypes.modules.config          import PipelineConfig
from ppp_subtypes.modules.profiler        import Profiler
from ppp_subtypes.modules.genesets        import (PPP_GENESETS, PATHWAY_HINTS,
    get_all_ppp_genes, gene_to_genesets, geneset_overlap)
from ppp_subtypes.modules.data_loader     import generate_synthetic, load_data
from ppp_subtypes.modules.preprocessing   import (filter_low_expression,
    tmm_normalise, mad_variance_filter, preprocess)
from ppp_subtypes.modules.dim_reduction   import hdlss_reduce, embed_2d
from ppp_subtypes.modules.clustering      import (consensus_cluster,
    bootstrap_stability, select_optimal_k, assign_subtypes)
from ppp_subtypes.modules.characterisation import (marker_genes_mwu,
    geneset_enrichment, pathway_report)
from ppp_subtypes.modules.visualisation   import (plot_consensus_heatmaps,
    plot_embedding, plot_marker_heatmap, plot_k_selection,
    plot_geneset_scores, plot_compute_profile)
from ppp_subtypes.modules.reporter        import write_report
from ppp_subtypes.main                    import run

# ── Shared fixtures (built once) ─────────────────────────────────────────────
def _tiny_cfg(out_dir):
    return PipelineConfig(
        synthetic_n_samples=24, synthetic_n_genes=500, synthetic_n_subtypes=3,
        min_count_threshold=2, min_samples_expressed=0.10, normalisation="tmm",
        top_var_genes=200, n_components=8, ledoit_wolf_shrinkage=True,
        low_resource_mode=True, k_range=[2,3], n_iterations=20,
        subsample_rate=0.70, bootstrap_ci=True, bootstrap_n=10,
        stability_threshold=0.30, use_ppp_genesets=True, geneset_weighting=2.0,
        profile_runtime=True, marker_top_n=10, out_dir=out_dir,
        write_report=True, save_intermediates=False, dpi=72)

_TMPDIR       = tempfile.mkdtemp(prefix="ppp_test_")
_CFG          = _tiny_cfg(_TMPDIR)
_RAW, _TLBLS  = generate_synthetic(_CFG)
_PROC         = preprocess(_RAW, _CFG)
_COORDS       = hdlss_reduce(_PROC, _CFG)
_MATS         = consensus_cluster(_COORDS, _CFG)
_OPT_K, _MET  = select_optimal_k(_MATS, _COORDS, _CFG)
_SUBS         = assign_subtypes(_COORDS, _OPT_K, _PROC.columns)
_MARKS        = marker_genes_mwu(_PROC, _SUBS, _CFG)
_ENRICH       = geneset_enrichment(_PROC, _SUBS)
_PW           = pathway_report(_SUBS, _MARKS)
_EMBED        = embed_2d(_COORDS, _CFG)

# =============================================================================
class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = PipelineConfig()
        self.assertEqual(c.synthetic_n_samples, 40)
        self.assertTrue(c.low_resource_mode)
    def test_out_path_created(self):
        with tempfile.TemporaryDirectory() as d:
            c = PipelineConfig(out_dir=str(Path(d)/"r"))
            self.assertTrue(c.out_path.exists())
    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            c = PipelineConfig(synthetic_n_samples=55, k_range=[2,3,4])
            p = str(Path(d)/"c.json")
            c.to_json(p); c2 = PipelineConfig.from_json(p)
            self.assertEqual(c2.synthetic_n_samples, 55)
            self.assertEqual(c2.k_range, [2,3,4])
    def test_nested_json(self):
        with tempfile.TemporaryDirectory() as d:
            blob = {"data":{"synthetic_n_samples":60},"cl":{"n_iterations":150}}
            p = Path(d)/"n.json"; p.write_text(json.dumps(blob))
            c = PipelineConfig.from_json(str(p))
            self.assertEqual(c.synthetic_n_samples, 60)
            self.assertEqual(c.n_iterations, 150)
    def test_summary_string(self):
        self.assertIsInstance(PipelineConfig().summary(), str)

class TestProfiler(unittest.TestCase):
    def test_records_timing(self):
        pr = Profiler(active=True)
        with pr("s"): time.sleep(0.01)
        df = pr.summary()
        self.assertEqual(df.iloc[0]["stage"], "s")
        self.assertGreaterEqual(df.iloc[0]["time_s"], 0.01)
    def test_multiple_stages(self):
        pr = Profiler(active=True)
        for n in "abc":
            with pr(n): pass
        self.assertEqual(len(pr.summary()), 3)
    def test_save_creates_files(self):
        with tempfile.TemporaryDirectory() as d:
            pr = Profiler(active=True)
            with pr("x"): pass
            pr.save(Path(d), dpi=72)
            self.assertTrue((Path(d)/"profiler_log.csv").exists())
            self.assertTrue((Path(d)/"profiler_timing.png").exists())
    def test_inactive_zero_ram(self):
        pr = Profiler(active=False)
        with pr("q"): pass
        self.assertEqual(pr.summary().iloc[0]["peak_ram_mb"], 0.0)

class TestGenesets(unittest.TestCase):
    def test_four_sets(self):       self.assertEqual(len(PPP_GENESETS), 4)
    def test_hints_match(self):     self.assertEqual(set(PATHWAY_HINTS), set(PPP_GENESETS))
    def test_unique_genes(self):
        g = get_all_ppp_genes()
        self.assertEqual(len(g), len(set(g)))
    def test_enough_genes(self):    self.assertGreater(len(get_all_ppp_genes()), 50)
    def test_gene_lookup_known(self):
        self.assertIn("Neuroinflammatory", gene_to_genesets("IL6"))
    def test_gene_lookup_unknown(self): self.assertEqual(gene_to_genesets("FAKE999"), [])
    def test_overlap(self):
        r = geneset_overlap(["IL6","ESR1","RANDOM"])
        self.assertIn("Neuroinflammatory", r); self.assertIn("IL6", r["Neuroinflammatory"])
    def test_overlap_empty(self):   self.assertEqual(geneset_overlap([]), {})

class TestDataLoader(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(_RAW.shape, (_CFG.synthetic_n_genes, _CFG.synthetic_n_samples))
    def test_label_length(self):    self.assertEqual(len(_TLBLS), _CFG.synthetic_n_samples)
    def test_no_negatives(self):    self.assertTrue((_RAW.values >= 0).all())
    def test_label_names_valid(self):
        for l in _TLBLS: self.assertIn(l, PPP_GENESETS)
    def test_hdlss(self):
        p, n = _RAW.shape
        self.assertGreater(p, n)
    def test_load_data(self):
        with tempfile.TemporaryDirectory() as d:
            e, l = load_data(_tiny_cfg(d))
            self.assertIsNotNone(l)

class TestPreprocessing(unittest.TestCase):
    def test_filter_reduces(self):
        f = filter_low_expression(_RAW, 5, 0.2)
        self.assertLessEqual(f.shape[0], _RAW.shape[0])
    def test_tmm_no_neg(self):
        self.assertTrue((tmm_normalise(_RAW.clip(lower=0)).values >= 0).all())
    def test_tmm_shape(self):
        self.assertEqual(tmm_normalise(_RAW).shape, _RAW.shape)
    def test_mad_top_n(self):
        n = tmm_normalise(_RAW)
        r = mad_variance_filter(n, 50, get_all_ppp_genes(), 2.0)
        self.assertEqual(r.shape[0], min(50, n.shape[0]))
    def test_ppp_weighting(self):
        n = tmm_normalise(_RAW); pg = get_all_ppp_genes()
        w = mad_variance_filter(n, 150, pg, 2.0)
        wo = mad_variance_filter(n, 150, [], 1.0)
        self.assertGreaterEqual(sum(g in pg for g in w.index),
                                sum(g in pg for g in wo.index))
    def test_proc_shape(self):
        self.assertEqual(_PROC.shape, (_CFG.top_var_genes, _CFG.synthetic_n_samples))
    def test_proc_no_nan(self):
        self.assertFalse(_PROC.isnull().any().any())

class TestDimReduction(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(_COORDS.shape, (_CFG.synthetic_n_samples, _CFG.n_components))
    def test_standardised(self):
        self.assertTrue(np.allclose(_COORDS.mean(0), 0, atol=1e-6))
        self.assertTrue(np.allclose(_COORDS.std(0),  1, atol=0.1))
    def test_no_nan(self):     self.assertFalse(np.isnan(_COORDS).any())
    def test_embed_shape(self):
        self.assertEqual(embed_2d(_COORDS, _CFG).shape, (_CFG.synthetic_n_samples, 2))

class TestClustering(unittest.TestCase):
    def test_matrix_shape(self):
        n = _COORDS.shape[0]
        for k, m in _MATS.items():
            self.assertEqual(m.shape, (n, n))
    def test_matrix_range(self):
        for k, m in _MATS.items():
            self.assertGreaterEqual(m.min(), -1e-6)
            self.assertLessEqual(m.max(), 1+1e-6)
    def test_symmetric(self):
        for k, m in _MATS.items():
            self.assertTrue(np.allclose(m, m.T, atol=1e-5))
    def test_diagonal_ones(self):
        for k, m in _MATS.items():
            self.assertTrue(np.allclose(np.diag(m), 1.0))
    def test_stability_range(self):
        s = bootstrap_stability(_COORDS, 2, _CFG)
        self.assertGreaterEqual(s, 0.0); self.assertLessEqual(s, 1.0)
    def test_optimal_k_in_range(self):  self.assertIn(_OPT_K, _CFG.k_range)
    def test_metrics_keys(self):
        for k in _CFG.k_range:
            for key in ("auc","silhouette","delta_auc","stability"):
                self.assertIn(key, _MET[k])
    def test_subtypes_length(self):     self.assertEqual(len(_SUBS), _CFG.synthetic_n_samples)
    def test_subtypes_format(self):
        for l in _SUBS: self.assertTrue(l.startswith("Subtype_"))

class TestCharacterisation(unittest.TestCase):
    def test_required_cols(self):
        req = {"subtype","gene","effect_size","p_value","is_ppp_gene"}
        self.assertTrue(req.issubset(_MARKS.columns))
    def test_markers_per_subtype(self):
        for st in _SUBS.unique():
            if (_SUBS == st).sum() >= 2:
                self.assertGreater((_MARKS["subtype"]==st).sum(), 0)
    def test_effect_size_range(self):
        self.assertTrue(_MARKS["effect_size"].between(-1,1).all())
    def test_enrichment_shape(self):
        self.assertEqual(_ENRICH.shape[0], len(PPP_GENESETS))
        self.assertEqual(_ENRICH.shape[1], _SUBS.nunique())
    def test_enrichment_cols(self):
        for st in _SUBS.unique(): self.assertIn(st, _ENRICH.columns)
    def test_pathway_report_keys(self):
        for st in _SUBS.unique():
            self.assertIn(st, _PW); self.assertIsInstance(_PW[st], list)

class TestVisualisation(unittest.TestCase):
    def _p(self, n): return Path(_TMPDIR)/n
    def _ok(self, p): self.assertTrue(p.exists()); self.assertGreater(p.stat().st_size, 0)
    def test_consensus(self):
        p = self._p("v_ch.png")
        plot_consensus_heatmaps(_MATS, _OPT_K, p, _CFG); self._ok(p)
    def test_embedding(self):
        p = self._p("v_emb.png")
        plot_embedding(_EMBED, _SUBS, None, p, _CFG); self._ok(p)
    def test_embedding_true_labels(self):
        p = self._p("v_embt.png")
        plot_embedding(_EMBED, _SUBS, _TLBLS, p, _CFG); self._ok(p)
    def test_marker_heatmap(self):
        p = self._p("v_mh.png")
        plot_marker_heatmap(_PROC, _SUBS, _MARKS, p, _CFG); self._ok(p)
    def test_k_selection(self):
        p = self._p("v_ks.png")
        plot_k_selection(_MET, _OPT_K, p, _CFG); self._ok(p)
    def test_geneset_scores(self):
        p = self._p("v_gs.png")
        plot_geneset_scores(_ENRICH, p, _CFG); self._ok(p)
    def test_compute_profile(self):
        p = self._p("v_cp.png")
        df = pd.DataFrame({"stage":["a","b"],"time_s":[1.,2.],"peak_ram_mb":[50.,80.]})
        plot_compute_profile(df, p, _CFG); self._ok(p)

class TestReporter(unittest.TestCase):
    def _run(self, suffix=""):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/f"r{suffix}.txt"
            pf  = pd.DataFrame({"stage":["s"],"time_s":[1.],"peak_ram_mb":[30.]})
            write_report(_CFG, _SUBS, _MET, _OPT_K, _MARKS, _PW, _ENRICH, pf, out)
            return out, out.read_text(encoding="utf-8")
    def test_created(self):
        import tempfile
        d = tempfile.mkdtemp()
        out = Path(d) / "rc.txt"
        import pandas as pd
        pf  = pd.DataFrame({"stage":["s"],"time_s":[1.],"peak_ram_mb":[30.]})
        write_report(_CFG,_SUBS,_MET,_OPT_K,_MARKS,_PW,_ENRICH,pf,out)
        self.assertTrue(out.exists())
    def test_nonempty(self):    _, t = self._run("n"); self.assertGreater(len(t), 100)
    def test_all_sections(self):
        _, t = self._run("s")
        for s in ["CONFIGURATION","SUBTYPE DISCOVERY","TOP MARKER GENES",
                  "PPP PATHWAY ANNOTATIONS","COMPUTE PROFILE"]:
            self.assertIn(s, t)
    def test_optimal_k_present(self):
        _, t = self._run("k"); self.assertIn(f"Optimal k = {_OPT_K}", t)

class TestIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = PipelineConfig(
                synthetic_n_samples=20, synthetic_n_genes=300,
                synthetic_n_subtypes=2, top_var_genes=100, n_components=6,
                k_range=[2,3], n_iterations=15, bootstrap_n=8,
                stability_threshold=0.20, bootstrap_ci=True,
                low_resource_mode=True, use_ppp_genesets=True,
                profile_runtime=True, out_dir=d, write_report=True,
                save_intermediates=False, dpi=72)
            res = run(cfg)
            for k in ("subtypes","markers","enrichment","profile","optimal_k","metrics"):
                self.assertIn(k, res)
            self.assertEqual(len(res["subtypes"]), cfg.synthetic_n_samples)
            self.assertIn(res["optimal_k"], cfg.k_range)
            for fname in ["sample_subtypes.csv","marker_genes.csv",
                          "consensus_heatmaps.png","embedding.png",
                          "marker_heatmap.png","k_selection.png","report.txt"]:
                p = Path(d)/fname
                self.assertTrue(p.exists(), f"Missing: {fname}")
                self.assertGreater(p.stat().st_size, 0, f"Empty: {fname}")

    def test_ram_under_500mb(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = PipelineConfig(
                synthetic_n_samples=20, synthetic_n_genes=300,
                k_range=[2,3], n_iterations=10, bootstrap_n=5,
                bootstrap_ci=False, profile_runtime=True, out_dir=d,
                write_report=False, save_intermediates=False, dpi=72)
            res = run(cfg)
            peak = res["profile"]["peak_ram_mb"].max()
            self.assertLess(peak, 500, f"Peak RAM {peak:.0f} MB > 500 MB")

if __name__ == "__main__":
    unittest.main(verbosity=2)