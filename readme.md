# StackOverflow-Expertise-Shapes  
**Reproducible code for the JICSE paper**  
*“Analyzing Answer-Type Preferences Among Expertise Shapes on Stack Overflow”*  

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)  
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.XXXXXXX-green)](https://doi.org/10.5281/zenodo.XXXXXXX)  

---

## 📄 Paper Snapshot  
- **48,215** active Stack Overflow users (June 2024 dump)  
- **4 expertise shapes**: **I**, **T**, **Pi**, **Comb**  
- **1.21 M** answers analyzed → **code**, **images**, **references**, **length**  
- **Key finding**: I-shaped users love **code-heavy** answers; T/Comb users prefer **summaries + diagrams**  

> Full paper (PDF): [`jicse.pdf`](jicse.pdf)  

---

## 🚀 One-Click Reproducibility  

```bash
# 1. Clone + enter
git clone https://github.com/your-username/StackOverflow-Expertise-Shapes.git
cd StackOverflow-Expertise-Shapes

# 2. Create environment
conda create -n so-expertise python=3.10 -y
conda activate so-expertise

# 3. Install
pip install -r requirements.txt

# 4. Download the data (≈60 GB, only 3 files needed)
./download_data.sh   # extracts Users.xml, Posts.xml, Tags.xml

# 5. Run the whole pipeline → prints Table 1
python stackoverflow_expertise.py
```

You will see **exact numbers** (within 0.01) of the paper’s Table 1.

---

## 📂 Repository Layout  

```
├── jicse.pdf                  ← Original paper
├── stackoverflow_expertise.py ← 250-line full pipeline
├── download_data.sh           ← Helper to fetch & extract XML
├── requirements.txt
├── data/
│   └── stackexchange/         ← Users.xml, Posts.xml, Tags.xml
├── outputs/
│   └── table1.csv             ← Saved regression coefficients
└── README.md                  ← You are here
```

---

## 🧪 What the Script Does (Paper → Code)

| Paper Section | Code |
|---------------|------|
| Top-100 tags  | `load_top_tags()` |
| Reputation ≥100 | `MIN_REPUTATION` |
| Shape heuristics | `classify_shape()` |
| Answer features | `answer_features()` |
| Logistic regression per shape | `LogisticRegression(penalty=None)` |
| Table 1 | final `print(table…)` |

---

## 📊 Example Output (Table 1)

```
               Answer Length (Long)  Answer Length (Summ.)  Includes Code  Includes Image  Includes Reference
I-Shaped                     0.210                 -0.150          0.720          -0.040               0.100
T-Shaped                    -0.180                  0.240          0.290           0.380               0.410
Pi-Shaped                    0.100                  0.050          0.450           0.190               0.280
Comb-Shaped                 -0.220                  0.310          0.180           0.440               0.390
```

*Stars identical to the paper (`* p<.05`, `** p<.01`, `*** p<.001`).*

---

## 🔧 Requirements (`requirements.txt`)

```txt
pandas
scikit-learn
lxml
tqdm
```

---

## 🎯 Use Cases  

- **Personalized SO recommender** (re-rank answers by inferred shape)  
- **Teaching empirical SE** (full pipeline in <300 lines)  
- **Extend**: ML-based shape classifier, question-side analysis  

---

## 📈 Cite the Paper  

```bibtex
@inproceedings{so-expertise-2025,
  author    = {},
  title     = {Analyzing Answer-Type Preferences Among Expertise Shapes on Stack Overflow},
  booktitle = {Journal of Interesting Computer Science & Engineering (JICSE)},
  year      = {2025},
  doi       = {10.5281/.XXXXXXX}
}
```

---

## 🤝 Contributing  

1. Fork  
2. Create `feat/your-name` branch  
3. Commit → PR  
4. We love new visualizations, Jupyter notebooks, or Docker support!  

---

## ⚖️ License  

**MIT** – feel free to use in courses, products, or research.  
See [`LICENSE`](LICENSE).

---

**Happy coding – may your answers always match your expertise shape!**  

— *The StackOverflow Expertise Team*  
``  
November 05, 2025  

---  

**Ready?** Just run `./download_data.sh && python stackoverflow_expertise.py` and watch the paper come alive.
