import json
import pandas as pd
from gaia_scorer import question_scorer


def analyze_hle_results(jsonl_file):
    """Analyze HLE results and show comparison between predictions and true answers"""
    # Read the results file
    results = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)

    print(f"Total questions: {len(df)}")
    print(f"Questions with predictions: {len(df[df['prediction'].notna()])}")
    print(f"Questions with errors: {len(df[df['agent_error'].notna()])}")

    # Calculate accuracy
    correct = 0
    total = 0

    print("\n=== DETAILED COMPARISON ===")
    print("Task ID | Prediction | True Answer | Correct | Score")
    print("-" * 60)

    for _, row in df.iterrows():
        if pd.isna(row['prediction']) or pd.isna(row['true_answer']):
            continue
        prediction = str(row['prediction'])
        true_answer = str(row['true_answer'])
        # Skip if true answer is "?" (test set)
        if true_answer == "?":
            continue
        # Score the prediction
        is_correct = question_scorer(prediction, true_answer)
        if is_correct:
            correct += 1
        total += 1
        print(f"{row['task_id'][:8]}... | {prediction[:20]:<20} | {true_answer[:20]:<20} | {is_correct} | {question_scorer(prediction, true_answer)}")

    # Calculate final accuracy
    if total > 0:
        accuracy = correct / total * 100
        print(f"\n=== FINAL RESULTS ===")
        print(f"Correct answers: {correct}/{total}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("No valid predictions to evaluate")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hle_analyzer.py <results_jsonl_file>")
    else:
        analyze_hle_results(sys.argv[1]) 