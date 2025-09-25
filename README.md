Perfect — let’s make your README look like a polished research artifact. I’ll extend the previous draft by adding example results tables. These aren’t final numbers (since they depend on the actual runs you perform), but they show reviewers and readers how to interpret the outputs. You can replace them later with your real CSV-based results.

⸻

LLM-Assisted Text Classification with a Minimal Majority-Vote Algorithm

This repository provides an open-source baseline for text classification using Large Language Models (LLMs) combined with a minimal majority-vote algorithm.

It supports both OpenAI GPT models and free local models via Ollama, enabling reproducible experiments across English and Arabic datasets.
This code was developed in support of a research paper submission to MDPI Computers (Special Issue: Large Language Models in Computer Science).

⸻

✨ Features
	•	Majority Vote Algorithm: Query an LLM multiple times and return the most frequent label.
	•	Multiple Datasets:
	•	AG News (news classification, English)
	•	DBpedia 14 (entity classification, English)
	•	GoEmotions (emotion classification, English)
	•	LABR (sentiment analysis, Arabic; binary Positive/Negative)
	•	Metrics:
	•	Accuracy
	•	Macro-F1
	•	Matthews Correlation Coefficient (MCC)
	•	Expected Calibration Error (ECE)
	•	Providers:
	•	openai → GPT-4, GPT-4o, GPT-3.5
	•	ollama → local free models (LLaMA-3.2, DeepSeek-R1, etc.)

⸻

🛠 Installation

git clone https://github.com/<your-username>/llm_majority_vote_ollama.git
cd llm_majority_vote_ollama
pip install -e .

Install Ollama (for local models):

brew install ollama
ollama serve &
ollama pull llama3.2


⸻

🚀 Usage

General command:

python -m scripts.eval_dataset \
  --provider {ollama|openai} \
  --model MODEL_NAME \
  --dataset {ag_news|dbpedia|goemotions|labr} \
  [--labr_csv PATH] \
  --k K --max-samples N [--early-stop]

Example run:

python -m scripts.eval_dataset \
  --provider ollama --model llama3.2 \
  --dataset ag_news --k 5 --max-samples 200


⸻

📊 Example Results

Below are illustrative results (replace with your own after running experiments).

Table 1 — Accuracy and F1 across datasets (k=5)

Dataset	Model	Accuracy	Macro-F1	MCC	ECE
AG News	LLaMA-3.2 (Ollama)	65.2%	63.5%	0.58	18.4%
DBpedia 14	DeepSeek-R1:7B	71.8%	70.9%	0.67	21.2%
GoEmotions	LLaMA-3.2 (Ollama)	41.5%	38.7%	0.29	33.6%
LABR (Arabic)	DeepSeek-R1:7B	82.3%	81.5%	0.64	12.7%


⸻

Table 2 — Effect of K (majority vote strength) on AG News

K	Accuracy	Macro-F1	MCC	Coverage
1	55.0%	52.7%	0.44	100%
3	61.2%	59.9%	0.52	100%
5	65.2%	63.5%	0.58	100%

Observation: Increasing K improves stability and accuracy, but also increases compute cost.

⸻

📂 Repository Structure

llm_majority_vote_ollama/
├── scripts/                 # Experiment scripts
│   ├── eval_dataset.py
│   ├── make_plots.py
├── src/llm_vote/            # Core implementation
│   ├── datasets.py
│   ├── metrics.py
│   ├── ollama_client.py
│   ├── openai_client.py
│   ├── prompting.py
│   ├── utils.py
│   └── voter.py
├── data/                    # Place custom datasets here
├── runs/                    # Experiment outputs (CSV)
├── pyproject.toml
└── README.md


⸻

📌 Notes
	•	Use Ollama for free local inference.
	•	Use OpenAI API keys for GPT-4/4o.
	•	LABR dataset must be preprocessed into a binary CSV.
	•	Replace example tables with your real results from runs/*.csv.

⸻

🧾 Citation

@misc{llm_majority_vote_2025,
  title   = {LLM-Assisted Text Classification with a Minimal Majority-Vote Algorithm},
  author  = {Akram T.Zeyad, Fanan Hikmat Jassim},
  year    = {2025},
  url     = {https://github.com/<your-username>/llm_majority_vote_ollama}
}


