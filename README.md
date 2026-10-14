# RACM-Forecast

**Regime-Adaptive Causal Multimodal Forecasting with LLM-Driven Dynamic Financial Knowledge Graphs and Conformal Uncertainty**

A modular end-to-end financial forecasting framework that combines multi-modal data (market, news, fundamentals, macro) with causal graph learning, regime detection, and conformal prediction.

## Architecture

```mermaid
flowchart TD
    subgraph Data["1. Multi-Modal Ingestion (Module 1)"]
        MKT["Market OHLCV Data"]
        NEWS["Financial News Texts"]
        FUND["Fundamental Ratios"]
        MACRO["Macro Indicators"]
    end

    subgraph Encoders["2. Multi-Modal Encoders (Modules 2 & 3)"]
        TE["Risk-Adaptive Temporal Encoder<br/>(Transformer / TCN)"]
        EE["LLM Event Extractor<br/>(OpenAI / Llama + NER)"]
        FE["Fundamental TabTransformer"]
        ME["Macro Temporal Encoder"]
    end

    subgraph Graph["3. Graph & Causal Discovery (Modules 4 & 5)"]
        KG["Dynamic Financial Knowledge Graph<br/>(NetworkX Snapshots + PyG Embeddings)"]
        CD["Causal DAG Discovery<br/>(Continuous NOTEARS Optimization)"]
    end

    subgraph Fusion["4. Graph Transformer & Regime-Conditioned MoE (Modules 6, 7, 8)"]
        GT["Causal Temporal Graph Transformer<br/>(RACM-GT Attention)"]
        REG["Market Regime Detector<br/>(Gaussian HMM: Bull, Bear, Volatile, Crisis)"]
        MOE["Regime-Conditioned Mixture-of-Experts<br/>(Specialist Neural Networks)"]
    end

    subgraph Predictions["5. Predictions & Uncertainty (Modules 9 & 10)"]
        RET["Return Regression Head"]
        DIR["Direction Classification Head (Up/Down/Flat)"]
        VOL["Volatility Head (Non-Negative Softplus)"]
        UQ["Conformal Uncertainty Quantification<br/>(Distribution-Free 90% Bounds & Adaptive ACI)"]
    end

    subgraph Explain["6. Explainability & Risk Analysis (Module 11)"]
        XAI["SHAP Attributions · Graph Explainer · Counterfactual What-Ifs"]
    end

    MKT --> TE
    NEWS --> EE
    FUND --> FE
    MACRO --> ME

    EE --> KG
    TE & FE & ME --> CD
    CD --> KG
    KG & TE --> GT
    MKT & MACRO --> REG
    GT & REG --> MOE
    MOE --> RET & DIR & VOL
    RET & VOL --> UQ
    MOE & GT --> XAI
```

![RACM-Forecast Framework](docs/architecture.png)

## Modules

| # | Module | Status |
|---|--------|--------|
| 1 | Multi-Modal Data Preprocessing | ✅ Done |
| 2 | Risk-Adaptive Temporal Encoder | ✅ Done |
| 3 | LLM-based Event Extraction | ✅ Done |
| 4 | Dynamic Financial Knowledge Graph | ✅ Done |
| 5 | Causal Graph Learning | ✅ Done |
| 6 | Causal Temporal Graph Transformer (RACM-GT) | ✅ Done |
| 7 | Market Regime Discovery | ✅ Done |
| 8 | Regime-Conditioned Mixture-of-Experts | ✅ Done |
| 9 | Prediction Heads | ✅ Done |
| 10 | Uncertainty Quantification | ✅ Done |
| 11 | Explainability and Risk Analysis | ✅ Done |

## Project Structure

```
racm-forecast/
├── config/              # YAML configs
├── data/                # Raw and processed data
├── preprocessing/       # Module 1 — loaders, alignment, features
├── encoders/            # Modules 2 & 3 — temporal, event extraction
├── knowledge_graph/     # Module 4 — graph construction & embedding
├── causal/              # Modules 5 & 6 — causal learning & RACM-GT
├── regime/              # Modules 7 & 8 — regime detection & MoE
├── prediction/          # Modules 9 & 10 — heads & uncertainty
├── explainability/      # Module 11 — SHAP, GNN, counterfactual
├── tests/               # Unit & integration tests
├── pipeline.py          # End-to-end pipeline
└── requirements.txt     # Dependencies
```

## Setup

```bash
git clone https://github.com/<your-username>/racm-forecast.git
cd racm-forecast
pip install -r requirements.txt
```

## Run Tests

```bash
python -m pytest tests/ -v
```

## License

MIT
