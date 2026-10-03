"""Evaluation helpers for ROUGE and BLEU."""

def compute_rouge_bleu(predictions, references):
    import evaluate
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
    return {**rouge_scores, "bleu": bleu_score["score"] / 100.0}
