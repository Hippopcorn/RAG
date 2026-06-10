import json
from pydantic import BaseModel


class Evaluator(BaseModel):
    """Compute retrieval quality metrics (Recall@k with IoU-based matching)
    by comparing student predictions to a ground-truth dataset."""

    def calculate_iou(self, start1: int, end1: int,
                      start2: int, end2: int) -> float:
        """Return the Intersection-over-Union of two character ranges, i.e.
        the proportion of overlapping characters relative to the union of
        the two source spans."""
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)

        intersection = max(0, intersection_end - intersection_start)
        union = max(end1, end2) - min(start1, start2)

        if union == 0:
            return 0.0
        return intersection / union

    def evaluate(self, student_answer_path: str,
                 dataset_path: str, k: int = 5) -> float:
        """Compare student predictions against the ground truth: a question is
        counted as successful when at least one of the top-``k`` retrieved
        sources shares its file path with a ground-truth source and has an
        IoU of at least 5% with it. Prints and returns the Recall@k (in %)."""
        with open(student_answer_path, encoding="utf-8") as f:
            student_data = json.load(f)
        predictions = student_data.get("search_results", [])

        with open(dataset_path, encoding="utf-8") as f:
            ground_truth_data = json.load(f)
        ground_truth_list = ground_truth_data.get("rag_questions", [])

        gt_dict = {q["question_id"]: q for q in ground_truth_list}

        total_questions = 0
        successful_retrievals = 0

        for pred in predictions:
            q_id = pred["question_id"]

            if q_id not in gt_dict:
                continue

            total_questions += 1
            true_sources = gt_dict[q_id]["sources"]
            predicted_sources = pred["retrieved_sources"][:k]

            question_success = False

            for true_src in true_sources:
                for pred_src in predicted_sources:
                    if true_src["file_path"] == pred_src["file_path"]:
                        iou = self.calculate_iou(
                            true_src["first_character_index"],
                            true_src["last_character_index"],
                            pred_src["first_character_index"],
                            pred_src["last_character_index"]
                        )
                        if iou >= 0.05:
                            question_success = True
                            break
                if question_success:
                    break
            if question_success:
                successful_retrievals += 1

        if total_questions == 0:
            print("No correspondence found"
                  " between the two files")
            return 0.0

        recall_at_k = (successful_retrievals / total_questions) * 100

        print("\n--- RAG EVALUATION ---")
        print(f"Nb query : {total_questions}")
        print(f"Query succeeded (IoU >= 5%) : {successful_retrievals}")
        print(f"Recall@{k} Final : {recall_at_k:.2f}%")

        return recall_at_k
