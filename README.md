# PoiRACG

## Overview
**PoiRACG** This repository contains the code for the paper "Covert Knowledge Poisoning Attacks in Retrieval-Augmented Code Generation"

## Key Innovations
### 🛡️ Dual-Phase Stealth Mechanism
1. **Vulnerability Feature Steganography**  
   - Constructs samples without detectable vulnerability patterns
   - Bypasses static analysis tools (Semgrep, Bandit)
   
2. **Knowledge Completion Inducement**  
   - Decomposes vulnerability logic into non-malicious fragments
   - Triggers model to introduce flaws during code generation
