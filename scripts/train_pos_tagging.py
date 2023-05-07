# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import classification_report
import argparse

def tokenize_and_align_labels(examples, column_names, tokenizer):
    tokenized_inputs = tokenizer(examples[column_names[0]], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[column_names[1]]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if "xx" in tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][i][word_idx]) else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def fine_tune_pos_tagging_model(train_file, dev_file, model_checkpoint, output_dir):
    # Define el nombre del dataset y la columna de las oraciones y las etiquetas POS:
    dataset_name = "custom_dataset"
    column_names = ("sentence", "labels")

    # Carga el tokenizador y el modelo pre-entrenado:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=NUM_POS_TAGS)

    # Carga los datos de entrenamiento y validación:
    train_dataset = load_dataset("text", data_files=train_file, split="train")
    dev_dataset = load_dataset("text", data_files=dev_file, split="train")

    # Tokeniza y alinea las etiquetas de los datos de entrenamiento y validación:
    tokenized_train = train_dataset.map(lambda examples: tokenize_and_align_labels(examples, column_names, tokenizer), batched=True)
    tokenized_dev = dev_dataset.map(lambda examples: tokenize_and_align_labels(examples, column_names, tokenizer), batched=True)

    # Entrenar el modelo
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                    'attention_mask': torch.stack([f[1] for f in data]),
                                    'labels': torch.stack([f[2] for f in data])}
        )

    eval_output = trainer.evaluate()

    # Imprimir las métricas de evaluación
    print("Accuracy: {:.2f}".format(eval_output['eval_accuracy']))
    precision, recall, f1, _ = eval_output['eval_precision_recall_fscore_support']
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    # Guardar el modelo entrenado
    trainer.save_model(output_dir)





def main():
    parser = argparse.ArgumentParser(description='Fine-tuning a POS tagging model.')
    parser.add_argument('--train_file', type=str, help='Path to the train file')
    parser.add_argument('--dev_file', type=str, help='Path to the dev file')
    parser.add_argument('--model_checkpoint', type=str, default='bert-base-cased', help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to the output directory')
    args = parser.parse_args()

    fine_tune_pos_tagging_model(args.train_file, args.dev_file, args.model_checkpoint, args.output_dir)

if __name__ == '__main__':
    main()