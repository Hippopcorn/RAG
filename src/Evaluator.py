import json
from pydantic import BaseModel


class Evaluator(BaseModel):

    def calculate_iou(self, start1: int, end1: int,
                      start2: int, end2: int) -> float:
        """ Get the communs index characters between source_1 and source_2,
            and return the percent of communs characters """
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)

        intersection = max(0, intersection_end - intersection_start)
        union = max(end1, end2) - min(start1, start2)

        if union == 0:
            return 0.0
        return intersection / union

    def evaluate(self, student_answer_path: str,
                 dataset_path: str, k: int = 5) -> float:
        # 1. Charger ton fichier de prédictions (généré par search_dataset)
        with open(student_answer_path, encoding="utf-8") as f:
            student_data = json.load(f)
        predictions = student_data.get("search_results", [])

        # 2. Charger le fichier de référence de l'école (AnsweredQuestions)
        with open(dataset_path, encoding="utf-8") as f:
            ground_truth_data = json.load(f)
        ground_truth_list = ground_truth_data.get("rag_questions", [])

        # Mettre la vérité terrain dans un dict indexé par question_id pour un accès O(1)
        gt_dict = {q["question_id"]: q for q in ground_truth_list}

        total_questions = 0
        successful_retrievals = 0

        # 3. Boucler sur tes prédictions pour vérifier la pertinence
        for pred in predictions:
            q_id = pred["question_id"]

            # Si la question n'existe pas dans la vérité terrain, on l'ignore
            if q_id not in gt_dict:
                continue

            total_questions += 1
            true_sources = gt_dict[q_id]["sources"]
            # On ne garde que les top-k sources retournées par ton BM25
            predicted_sources = pred["retrieved_sources"][:k]

            # Est-ce qu'au moins UNE des vraies sources est capturée par tes prédictions ?
            question_success = False

            for true_src in true_sources:
                for pred_src in predicted_sources:
                    # Étape cruciale : On ne compare que si c'est le même fichier !
                    if true_src["file_path"] == pred_src["file_path"]:
                        iou = self.calculate_iou(
                            true_src["first_character_index"],
                            true_src["last_character_index"],
                            pred_src["first_character_index"],
                            pred_src["last_character_index"]
                        )
                        # Si l'IoU est de 5% ou plus, la source est considérée comme trouvée !
                        if iou >= 0.05:
                            question_success = True
                            break  # Pas besoin de vérifier le reste pour cette vraie source
                if question_success:
                    break  # La question est validée, on passe à la question suivante

            if question_success:
                successful_retrievals += 1

        # 4. Calcul du score Recall@k final
        if total_questions == 0:
            print("Aucune correspondance trouvée entre les IDs des deux fichiers.")
            return 0.0

        recall_at_k = (successful_retrievals / total_questions) * 100

        print("\n--- ÉVALUATION RAG ---")
        print(f"Questions évaluées : {total_questions}")
        print(f"Requêtes réussies (IoU >= 5%) : {successful_retrievals}")
        print(f"Recall@{k} Final : {recall_at_k:.2f}%")

        return recall_at_k
