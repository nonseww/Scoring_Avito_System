from pathlib import Path
import csv


class EvaluationService:
    """Сервис, считающий итоговый Recall@50"""
    def recall(self, predictions: dict, gold: dict) -> float:
        query_recalls = []
        for qid, relevant in gold.items():
            pred_set = set(predictions.get(qid, [])[:50])
            gold_set = set(relevant)
            query_recalls.append(len(pred_set & gold_set) / len(gold_set))
        return sum(query_recalls) / len(query_recalls)

    def save_answer(self, predictions: dict, path: Path) -> None:
        """Сохранение ответа с кандидатами в файл"""
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "answer"])
            for qid, item_ids in predictions.items():
                answer = " ".join(item_ids[:50])
                writer.writerow([qid, answer])


if __name__ == "__main__":
    service = EvaluationService()
    pred = {"A": ["x1"], "B": ["y1"], "C": ["z1", "z2"]}
    gold = {"A": ["x1"], "B": ["y1", "y2"], "C": ["z9"]}
    print(service.recall(pred, gold))

    service.save_answer({"A": ["x1", "x2"], "B": ["y1"]}, Path("test.csv"))