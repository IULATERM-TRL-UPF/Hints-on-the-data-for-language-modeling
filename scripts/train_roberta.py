import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
from transformers import RobertaTokenizerFast
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM


def train_language_model(train_path, eval_path, language):

    tokenizer = Tokenizer(WordLevel())
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(vocab_size=52_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    tokenizer.train([train_path], trainer)
    tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )

    tokenizer.save("../"+str(language)+"/tokenizer.json")

    tokenizer.enable_truncation(max_length=512)

    config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    tokenizer_config = {"max_len": 512}

    with open("../"+str(language)+"/tokenizer_config.json", 'w') as fp:
        json.dump(tokenizer_config, fp)

    tokenizer = RobertaTokenizerFast.from_pretrained("../"+str(language), max_len=512)

    model = RobertaForMaskedLM(config=config)

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128,
    )

    dataset_eval = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=eval_path,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./"+str(language),
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_gpu_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        evaluation_strategy="steps",
        eval_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=dataset_eval
    )

    trainer.train()

    print(trainer.state.log_history[-2])

    trainer.save_model("./"+str(language))


def main():
    # Create the ArgumentParser object and define the arguments
    parser = argparse.ArgumentParser(description='Language Model Training')
    parser.add_argument('-train_file', '--input', help='Training data file path')
    parser.add_argument('-dev_file', '--evaluation', help='Evaluation data file path')
    parser.add_argument('-language', '--language', help='Language used')

    # Parse the command line arguments
    args = parser.parse_args()

    # Verify if input and evaluation paths, and language were provided
    if not args.input or not args.evaluation or not args.language:
        parser.error('Input, evaluation, and language paths must be provided')

    train_path = args.input
    eval_path = args.evaluation
    language = args.language

    # Call the language model training function
    train_language_model(train_path, eval_path, language)

if __name__ == "__main__":
    main()

