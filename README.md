# PoiRACG


This repository contains the code for the paper "Covert Knowledge Poisoning Attacks in Retrieval-Augmented Code Generation".

## Key Innovations
### 🛡️ Dual-Phase Stealth Mechanism
1. **Vulnerability Feature Steganography**  
   - Constructs samples without detectable vulnerability patterns
   - Bypasses static analysis tools (Semgrep, Bandit)
   
2. **Knowledge Completion Inducement**  
   - Decomposes vulnerability logic into non-malicious fragments
   - Triggers model to introduce flaws during code generation

## 🤗 Dataset & Evaluation
In this work, we use CoNaLa in for attack testing. The data for the experiment can be found in the dataset folder. [CoNaLa](https://huggingface.co/datasets/neulab/conala) can also be downloaded directly from huggingface.


## Project Structure
```
PoiRACG 
|-- get_poisoncode.py # implement the indution code generation
|-- run.py # implement the attack against RACG
|-- evaluate_cases.py # implement attack result evaluation 
```

## Execution Workflow
### Stage 1: Induction Code Generation
```
python get_poisoncode.py
```
Generates poisoned code fragments with hidden vulnerability patterns.

### Stage 2: RACG Attack Execution
```
python run.py
```
Runs the poisoning attack against Retrieval-Augmented Code Generation systems.

### Stage 3: Performance Evaluation
```
python evaluate_cases.py
```
executes the command to evaluate the performance results.
