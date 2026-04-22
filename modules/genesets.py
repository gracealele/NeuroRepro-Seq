

from __future__ import annotations

PPP_GENESETS: dict[str, list[str]] = {

    # ----- Subtype A: Neuro-inflammatory -------
    # Refs: Bergink 2015, Frey 2022
    # Elevated cytokines, microglial activation, blood-brain-barrier distruption
    
    "Neuroinflammatory": [
        # Pro-inflammatory cytokines
        "IL1B", "IL6", "IL8", "IL17A", "IL18", "IL12A", "IL12B", "TNFRSF1A", "IFNG", "IFNA1", "TNF",

        # Chemokines
        "CCL2", "CCL3", "CCL4", "CCL5", "CXCL10", "CXCL8", 

        # Complement / innate immune
        "C1QB", "C3", "C4A", "CFB", "SERPING1", "PTGS2", "HMGB1",

        # Microglial markers
        "AIFI", "TMEM119", "P2PRY12", "CX3CR1", "TREM2",

        # NF-kB pathway
         "NFKB1", "NFKBIA", "RELA", "IKBKB", "TLR4", "MYD88",  "STAT3",

        # Oxidative stress
        "NOX4", "SOD2", "HMOX1", "NRF2",

        # BBB distruption
        "MMP9", "MMP2", "OCLN", "CLDN5", "TJP1",

        # Acute phase
        "CRP", "SAA1", "ORM1", "FGA",
    ],
    
    # ── HPA Axis / Hormonal dysregulation ───────────────────────────────────
    # Refs: Deschamps 2016, Payne 2020, Osborne 2017
    
    "HPA_Hormonal": [
        "ESR1", "ESR2",
        "PRL",  "PRLR",
        "CRH",  "CRHR1", "CRHR2",
        "NR3C1","FKBP5", "HSD11B1", "HSD11B2",
        "OXTR", "OXT",
        "AVP",  "AVPR1A","AVPR1B",
        "POMC", "MC2R",
        "GNRH1","GNRHR", "KISS1",   "KISS1R",
    ],
    
    # ── Dopaminergic / Synaptic signalling ──────────────────────────────────
    # Refs: Bergink 2011, Forty 2014, Jones 2014
    
    "Dopaminergic_Synaptic": [
        "DRD1", "DRD2",  "DRD3",  "DRD4",
        "TH",   "SLC6A3","COMT",
        "MAOA", "MAOB",
        "BDNF", "NTRK2",
        "GRIN1","GRIN2A","GRIN2B",
        "GABRA1","GABRB2",
        "SLC6A4","HTR2A","HTR1A",
        "DISC1","NRG1",  "DTNBP1",
    ],
    
    # ── Immune-Oxidative stress ──────────────────────────────────────────────
    # Refs: Dahan 2021, Balan 2019
    "Immune_Oxidative": [
        "GSTP1","GSTM1",
        "SOD1", "SOD2",
        "CAT",  "GPX1",
        "NFE2L2","HMOX1",
        "NOS2", "NOS3",  "MPO",
        "CD4",  "CD8A",  "CD19",  "FOXP3",
        "NCAM1","B2M",
        "HLA-A","HLA-DRA",
        "CTLA4","PDCD1",
    ],
}

PATHWAY_HINTS: dict[str, list[str]] = {
    "Neuroinflammatory": [
        "Pro-inflammatory cytokine signalling (IL-1β / IL-6 / TNF-α)",
        "NF-κB pathway activation",
        "Complement system dysregulation (C3 / C4)",
        "Microglial activation and neuroinflammation",
        "Blood-brain barrier disruption via CXCL10",
    ],
        
    "HPA_Hormonal": [
        "HPA axis hyperactivation (CRH → cortisol)",
        "Oestrogen receptor signalling (ESR1 / ESR2)",
        "Prolactin / prolactin receptor pathway",
        "Oxytocin / vasopressin social bonding circuits",
        "Glucocorticoid receptor sensitivity (NR3C1 / FKBP5)",
    ],
    
    "Dopaminergic_Synaptic": [
        "Dopamine receptor dysregulation (DRD2 / DRD4)",
        "BDNF–TrkB neurotrophic signalling",
        "Glutamatergic NMDA receptor function (GRIN1 / GRIN2B)",
        "Serotonin reuptake and receptor binding (SLC6A4 / HTR2A)",
        "Dopamine synthesis and catabolism (TH / COMT / MAOA)",
    ],
    
    "Immune_Oxidative": [
        "Oxidative stress and antioxidant defence (SOD / CAT / GPX1)",
        "NRF2-mediated stress response",
        "T-cell regulatory dysfunction (FOXP3 / CTLA4 / PDCD1)",
        "MHC class I / II antigen presentation",
        "Nitric oxide signalling (NOS2 / NOS3)",
    ],
}    

def get_all_ppp_genes() -> list[str]:
    """Return a deduplicated flat list of all PPP signature genes."""
    seen: set[str] = set()
    genes: list[str] = []
    for gene_list in PPP_GENESETS.values():
        for g in gene_list:
            if g not in seen:
                seen.add(g)
                genes.append(g)
    return genes

def gene_to_genesets(gene: str) -> list[str]:
    """Return which PPP gene sets a given gene belongs to."""
    return [name for name, lst in PPP_GENESETS.items() if gene in lst]


def geneset_overlap(gene_list: list[str]) -> dict[str, list[str]]:
    """
    Given an arbitrary gene list, return the PPP gene sets it overlaps with
    and the overlapping genes.
 
    Returns:
    dict: {geneset_name: [overlapping_genes]}
    """
    result: dict[str, list[str]] = {}
    gene_set = set(gene_list)
    for name, ppp_genes in PPP_GENESETS.items():
        overlap = [g for g in ppp_genes if g in gene_set]
        if overlap:
            result[name] = overlap
    return result