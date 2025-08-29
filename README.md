# 🚗💥 From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios
<div align="center">
<a href="https://arxiv.org/abs/2502.02145"><img src="https://img.shields.io/badge/Paper-PDF-red.svg" alt="Paper Badge"/></a>
<a href="https://github.com/TUM-AVS/From-Words-to-Collisions/stargazers"><img src="https://img.shields.io/github/stars/TUM-AVS/From-Words-to-Collisions" alt="Stars Badge"/></a>
<a href="https://github.com/TUM-AVS/From-Words-to-Collisions/network/members"><img src="https://img.shields.io/github/forks/TUM-AVS/From-Words-to-Collisions" alt="Forks Badge"/></a>
<a href="https://github.com/TUM-AVS/From-Words-to-Collisions/pulls"><img src="https://img.shields.io/github/issues-pr/TUM-AVS/From-Words-to-Collisions" alt="Pull Requests Badge"/></a>
<a href="https://github.com/TUM-AVS/From-Words-to-Collisions/issues"><img src="https://img.shields.io/github/issues/TUM-AVS/From-Words-to-Collisions" alt="Issues Badge"/></a>
<a href="https://github.com/TUM-AVS/From-Words-to-Collisions/blob/main/LICENSE"><img src="https://img.shields.io/github/license/TUM-AVS/From-Words-to-Collisions" alt="License Badge"/></a>
</div>

![Framework](https://github.com/user-attachments/assets/a3253a9d-f6a3-4969-a3ad-25406957537f)

We propose a novel framework that leverages Large Language Models (LLMs) for:
- 🧠 **Evaluation**: Assessing safety-criticality of driving scenarios with use cases: Scenario evaluation and Safety inference.
- 🛠️ **Generation**: Adversarially generating safety-critical scenarios with controllable agent trajectories.

---
## ✨ Highlights

<table>
  <tr>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/7f82641d-b244-4b71-927a-bb2793f0dbc5" alt="IEEE-ITSC-original" width="100%">
        <figcaption align="center">🟢 Original Scenario</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/694337df-1a7f-4679-831e-e0af19ecc410" alt="IEEE-ITSC-modified" width="100%">
        <figcaption align="center">🔴 Modified Scenario (Collision)</figcaption>
      </figure>
    </td>
  </tr>
</table>

## :fire: Updates
- [July 2025] Paper accepted at IEEE ITSC 2025
- [May 2025] Project repository initialized
---
## 🗂️ Project Structure

### 📁 `LLM/` – LLM-Based Analysis Scripts
- `agent/`: Analyze agent-based scenarios
- `agent_normal/`: Analyze normal agent behaviors
- `scenario/`: Analyze collision scenarios
- `scenario_normal/`: Analyze normal driving scenarios

### 📁 `Generation/` – Scenario Generation & Safety Metrics
- `BEL_Antwerp-1_14_T-1/`: Original normal scenarios
- `BEL_Antwerp-1_14_T-1n/`: Generated adversarial scenarios
- `Metrices/`: Safety metric for these two scenarios

### 📁 `Data_collection/` – Trajectory Data Tools
- `Trajectory_collection/`: Collect vehicle trajectories
- `Riskscore_calculation/`: Compute risk scores
- `Safety_metrics_collection/`: Extract safety metrics
- `CloesdID_identification/`: Identify nearby agents
- `generate_timestep_report.py`: Generate reports for each timestep

### 📁 `Scenarios/` – Dataset of Driving Scenarios
- `normal_scenarios/`: 100 normal scenarios (Frenetix planner)
- `collision_scenarios/`: 100 collision scenarios (Frenetix planner)

### 📁 `Results/` – Analysis & Validation
- `LLM/`: Results from LLM evaluations
- `output_validation/`: Validation for collision scenarios
- `output_validation_normal/`: Validation for normal scenarios

## ⚙️ Setup & Configuration

### ✅ Requirements
- Python 3.8+
- [CommonRoad](https://commonroad.in.tum.de/)
- [Frenetix Motion Planner](https://github.com/TUM-AVS/Frenetix-Motion-Planner/tree/main)

### 🔐 API Keys
Create a `.env` file in the root directory with your API keys:
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
DEEPSEEK_API_KEY=your_deepseek_key

### 📖 Citation
If you find this work helpful in your research, please consider citing us:

@article{gao2025words,
  title={From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios},
  author={Gao, Yuan and Piccinini, Mattia and Moller, Korbinian and Alanwar, Amr and Betz, Johannes},
  journal={arXiv preprint arXiv:2502.02145}  [Add to Citavi project by ArXiv ID] ,
  year={2025}
}







