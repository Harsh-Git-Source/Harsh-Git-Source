# Architecture

## BiLSTM path

`CNN/DailyMail article -> cleaning -> NLTK tokenization -> vocabulary -> frozen GloVe -> BiLSTM encoder -> Luong attention -> LSTM decoder -> beam search -> summary`

## FLAN-T5 path

`CNN/DailyMail article -> cleaning -> "summarize:" prefix -> T5 tokenizer -> FLAN-T5 encoder/decoder + LoRA adapters -> beam search -> summary`

## Evaluation

Both approaches are evaluated using ROUGE-1, ROUGE-2, ROUGE-L and BLEU in the supplied project implementations.
