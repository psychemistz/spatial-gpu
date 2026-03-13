"""Tests for reference data loading."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatialgpu.deconvolution.reference import (
    get_cancer_signature,
    load_cancer_dictionary,
    load_comb_ref,
    load_gene_set,
    load_lr_database,
    load_ref_normal_lihc,
    read_gmt,
    write_gmt,
)


class TestCombRef:
    def test_load_comb_ref_structure(self):
        ref = load_comb_ref()
        assert "refProfiles" in ref
        assert "sigGenes" in ref
        assert "lineageTree" in ref

    def test_ref_profiles_shape(self):
        ref = load_comb_ref()
        profiles = ref["refProfiles"]
        assert isinstance(profiles, pd.DataFrame)
        assert profiles.shape[0] > 15000  # >15k genes
        assert profiles.shape[1] > 20  # >20 cell types

    def test_sig_genes_not_empty(self):
        ref = load_comb_ref()
        assert len(ref["sigGenes"]) > 0
        for key, genes in ref["sigGenes"].items():
            assert isinstance(genes, list)

    def test_lineage_tree_hierarchy(self):
        ref = load_comb_ref()
        tree = ref["lineageTree"]
        assert len(tree) > 5  # >5 major lineages
        for key, subtypes in tree.items():
            assert isinstance(subtypes, list)
            assert len(subtypes) >= 1

    def test_macrophage_subtypes(self):
        ref = load_comb_ref()
        tree = ref["lineageTree"]
        assert "Macrophage" in tree
        mac_subtypes = tree["Macrophage"]
        assert len(mac_subtypes) >= 2  # M1, M2 at minimum


class TestCancerDictionary:
    def test_load_structure(self):
        cd = load_cancer_dictionary()
        assert "CNA" in cd
        assert "expr" in cd

    def test_cna_signatures(self):
        cd = load_cancer_dictionary()
        assert len(cd["CNA"]) > 0
        for name, sig in cd["CNA"].items():
            assert isinstance(sig, pd.Series)
            assert len(sig) > 0

    def test_expr_signatures(self):
        cd = load_cancer_dictionary()
        assert len(cd["expr"]) > 0

    def test_brca_exists(self):
        cd = load_cancer_dictionary()
        brca_found = any("BRCA" in name for name in cd["CNA"]) or any(
            "BRCA" in name for name in cd["expr"]
        )
        assert brca_found


class TestCancerSignature:
    def test_brca_auto(self):
        sig_type, sig = get_cancer_signature("BRCA")
        assert sig_type in ("CNA", "expr")
        assert len(sig) > 0

    def test_pancan(self):
        sig_type, sig = get_cancer_signature("PANCAN")
        assert sig_type == "expr"

    def test_seq_depth_fallback(self):
        sig_type, sig = get_cancer_signature("BRCA", sig_type="seq_depth")
        assert sig_type == "seq_depth"


class TestRefNormalLIHC:
    def test_load(self):
        ref = load_ref_normal_lihc()
        assert "refProfiles" in ref
        assert "sigGenes" in ref
        assert "lineageTree" in ref


class TestLRDatabase:
    def test_load(self):
        lr = load_lr_database()
        assert isinstance(lr, pd.DataFrame)
        assert lr.shape[0] > 2000  # ~2500 L-R pairs


class TestGMT:
    def test_read_hallmark(self):
        gmt = load_gene_set("Hallmark")
        assert len(gmt) >= 50  # MSigDB Hallmark has 50 sets
        for name, genes in gmt.items():
            assert len(genes) > 0

    def test_read_cancer_cell_state(self):
        gmt = load_gene_set("CancerCellState")
        assert len(gmt) > 0

    def test_read_tls(self):
        gmt = load_gene_set("TLS")
        assert len(gmt) > 0

    def test_write_roundtrip(self, tmp_path):
        original = {"SetA": ["GENE1", "GENE2", "GENE3"], "SetB": ["GENE4", "GENE5"]}
        path = tmp_path / "test.gmt"
        write_gmt(original, path)
        loaded = read_gmt(path)
        assert loaded == original
