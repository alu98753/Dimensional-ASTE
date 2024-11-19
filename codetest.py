def compute_metrics(pred_triplets, gold_triplets):
    pred_set = set(pred_triplets)
    gold_set = set(gold_triplets)

    # True positives
    true_positives = pred_set & gold_set
    print(f"Predicted Set: {pred_set}")
    print(f"Gold Set: {gold_set}")
    print(f"True Positives: {true_positives}")

    # Precision
    precision = len(true_positives) / len(pred_set) if len(pred_set) > 0 else 0.0
    print(f"Precision: {precision}")

    # Recall
    recall = len(true_positives) / len(gold_set) if len(gold_set) > 0 else 0.0
    print(f"Recall: {recall}")

    # F1-score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"F1 Score: {f1}")

    return precision, recall, f1

# 示例预测的三元组
pred_triplets = [
    ("Aspect1", "Opinion1", "Intensity1"),
    ("Aspect2", "Opinion2", "Intensity2"),
    ("Aspect3", "Opinion3", "Intensity3"),
]

# 示例真实的三元组
gold_triplets = [
    ("Aspect1", "Opinion1", "Intensity1"),  # 匹配
    ("Aspect2", "Opinion2", "Intensity2"),  # 匹配
    ("Aspect4", "Opinion4", "Intensity4"),  # 不匹配
]

print(compute_metrics(pred_triplets,gold_triplets))