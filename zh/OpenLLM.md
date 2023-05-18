## Open LLM Leaderboard

### With the plethora of large language models (LLMs) and chatbots being released week upon week, often with grandiose claims of their performance, it can be hard to filter out the genuine progress that is being made by the open-source community and which model is the current state of the art. The 🤗 Open LLM Leaderboard aims to track, rank and evaluate LLMs and chatbots as they are released. We evaluate models on 4 key benchmarks from the Eleuther AI Language Model Evaluation Harness , a unified framework to test generative language models on a large number of different evaluation tasks. A key advantage of this leaderboard is that anyone from the community can submit a model for automated evaluation on the 🤗 GPU cluster, as long as it is a 🤗 Transformers model with weights on the Hub. We also support evaluation of models with delta-weights for non-commercial licensed models, such as LLaMa.

## Evaluation is performed against 4 popular benchmarks:

### AI2 Reasoning Challenge (25-shot) - a set of grade-school science questions.
HellaSwag (10-shot) - a test of commonsense inference, which is easy for humans (~95%) but challenging for SOTA models.
MMLU (5-shot) - a test to measure a text model’s multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more.
TruthfulQA (0-shot) - a benchmark to measure whether a language model is truthful in generating answers to questions.
