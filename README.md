Search for a single query:
uv run python -m src search "query"



Search pour les questions de doc :

uv run python -m src search_dataset



Search pour les questions de code :

uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json



Evaluate avec la moulinette :

cd moulinette

./moulinette-ubuntu evaluate_student_search_results \
  --student_answer_path ../data/output/search_results/dataset_docs_public.json \
  --dataset_path ../data/datasets/AnsweredQuestions/dataset_docs_public.json



answer for a single query:

uv run python -m src answer "query"
