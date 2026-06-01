# Threatscope

**POC — A research-grade pipeline demonstrating data engineering and cybersecurity concepts, not production software.**
- Currently, the produced models are not trained on sufficient data to produce meaningful predictions. The architecture and pipeline are the artifact, not the weights.**
- This project currently doesn't consider distributability or dependency versions apart from the development environment.

**Threat Actor Behavior Prediction** – Ingest public cybersecurity threat intelligence, organize it into structured feature sets, and train a model to classify and predict threat actor tactics (e.g., mapping attacker behavior to MITRE ATT&CK techniques).

**Goal** – Assist security analysts in triaging incidents faster by auto-labeling observed behaviors.

---
# Design

### Dataflow

![Alt text description](./public/Threatscope_dataflow.png)

</br>

---

### Setup
**1. Download and enter the project directory**
```bash
# Clone the repo
git clone https://github.com/ZandtLavish/threatscope

# Enter the project directory
cd threatscope/

# Install the build tool & build the package
pip install build
python -m build

# Install `threatscope`
pip install -e .    # Editable mode suggested when making configuration changes

```
</br>

**2. Tailor `config.yaml` to setup API Keys, hyperparameters, etc.**
</br>

---

### Command Flow

`pipeline` → `train` → `evaluate` → `predict`

*NOTE – The MLflow commands have `demo` modes for you to run `train`/`evaluate`/`predict` end-to-end quickly with zero credentials*

**1. ETL**
```bash
threatscope pipeline
```
</br>

**2. Create and test your model**
```bash
threatscope train
threatscope evaluate
```
</br>

**3. Predict a tactic**
```bash
threatscope predict --description <CVE/Incident description>
```