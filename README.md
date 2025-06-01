# PoiRACG: Stealthy Poisoning Attack for Retrieval-Augmented Code Generation

## Overview
**PoiRACG** is the first knowledge base poisoning attack targeting Retrieval-Augmented Code Generation (RACG) systems. By contaminating retrieval knowledge bases with semantically suggestive code snippets, it induces LLMs to generate exploitable vulnerable code through contextual completion capabilities.

## Key Innovations
### 🛡️ Dual-Phase Stealth Mechanism
1. **Vulnerability Feature Steganography**  
   - Constructs samples without detectable vulnerability patterns
   - Bypasses static analysis tools (Semgrep, Bandit)
   
2. **Knowledge Completion Inducement**  
   - Decomposes vulnerability logic into non-malicious fragments
   - Triggers model to introduce flaws during code generation
