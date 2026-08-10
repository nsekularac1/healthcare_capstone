# System Architecture Diagram

The diagram below is architecture figure for the paper, README, and mentor presentation.

```mermaid
flowchart LR
    subgraph Clinical["Clinical Branch — Diabetes 130-US Hospitals"]
        D1[Raw clinical encounters] --> D2[Clinical preprocessing]
        D2 --> D3[Logistic Regression readmission model]
        D3 --> D4[Readmission probability + risk tier]
    end

    subgraph Wearable["Wearable Branch — PAMAP2"]
        P1[Raw wearable sensor files] --> P2[Sensor preprocessing + interpolation]
        P2 --> P3[100-step overlapping windows]
        P3 --> P4[PyTorch LSTM activity model]
        P4 --> P5[Activity label + confidence]
    end

    R[User request] --> G[Deterministic safeguards]
    G -->|Allowed| OAI[OpenAI tool-using agent]
    G -->|Out of scope| SF[Safety fallback]

    OAI -->|Function call| D3
    OAI -->|Function call| P4

    D4 --> OAI
    P5 --> OAI

    OAI --> X[Bounded generative explanation]
    X --> Y[Decision-support output]
    Y --> Z[Human review]

    DP[Data provenance boundary:
Datasets represent different populations.
No patient-level join.] -.-> D4
    DP -.-> P5
```

## Figure Caption

**Figure 1. Integrated Intelligent Health Monitoring and Clinical Decision Support System.** The architecture maintains separate clinical and wearable data pipelines, exposes each trained model as a bounded local tool, and uses an OpenAI-backed agent to select tools and synthesize structured outputs. The two datasets represent different populations; therefore, integration occurs only at the application layer and does not imply patient-level linkage or causation.
