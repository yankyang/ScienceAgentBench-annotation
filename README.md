# ScienceAgentBench-annotation

This repository showcases my annotation and evaluation work for the **ScienceAgentBench (SAB)** and **AutoSDT** projects — two large-scale efforts to build transparent and standardized scientific reasoning benchmarks for generalist AI agents.

---

## 🚀 My Contribution

- **Designed and implemented evaluation scripts** (`eval_programs/`) that verify agent outputs across diverse scientific domains, including survival analysis, geoscience, and biomedical modeling.  
  Each script defines reproducible evaluation metrics and data integrity checks for automatic benchmarking.

- **Developed and validated gold-standard programs** (`gold_programs/`) serving as the reference logic for SAB task correctness.  
  I ensured consistency between AutoSDT-generated code and benchmark expectations by refining program structure, handling exceptions, and improving readability.

- **Standardized data pipelines between AutoSDT and SAB**, enabling smooth execution of generated scripts within the evaluation framework.  
  This included preprocessing automation, directory structuring, and cross-system compatibility.

- **Led large-scale verification of benchmark tasks**, testing correctness across multiple datasets (e.g., METABRIC, MouseAtlas, Pancreas) and ensuring alignment between model outputs and gold results.

- **Optimized reproducibility and clarity**, restructuring the repository into lightweight, public-safe format (with heavy data moved to external storage).

---

## 🧩 Technical Focus

- Python (PyTorch, NumPy, Pandas, SciPy, Scanpy)  
- Automated evaluation scripting and task validation  
- Benchmark reproducibility and anti-contamination checks  
- Large-scale dataset handling under Git/GitHub constraints  
- Integration of AutoSDT-generated scientific task code with SAB evaluation layer  

---

## 📂 Repository Overview

ScienceAgentBench-annotation/
│
├── eval_programs/ # My implemented evaluation scripts for benchmark tasks
├── gold_programs/ # Verified gold-standard task logic
├── LICENSE
└── README.md


---

## 🔗 External Resources

Due to GitHub’s file size limits, large `.h5ad` and dataset files are hosted externally.  
Full benchmark data and outputs are available here:  
📦 [Google Drive Download Link](YOUR_LINK_HERE)

---

## 🧑‍💻 Author

**Yankai Yang**  
University of Wisconsin–Madison  
📧 yang693@wisc.edu  
🔗 [GitHub Profile](https://github.com/yankyang)

---

*This repository documents my hands-on contribution to the ScienceAgentBench benchmark and AutoSDT auto-annotation framework, focusing on reproducibility, correctness, and scientific AI evaluation.*
