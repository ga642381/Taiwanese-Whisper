import inspect
import random
import sys

import nlp2
from datasets import load_dataset, Audio
from transformers import Seq2SeqTrainer
from transformers import Trainer
from transformers import TrainingArguments, Seq2SeqTrainingArguments
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor

from module.args import parse_args
from module.data_processing import (
    encode_dataset,
    DataCollatorCTCWithPadding,
    prepare_dataset_hf,
    prepare_dataset_custom,
    prepare_dataset_whisper,
    DataCollatorSpeechSeq2SeqWithPadding,
)
from module.metric import cer_cal, wer_cal
from module.utility import FreezingCallback


def main(arg=None):
    input_arg, other_arg = parse_args(sys.argv[1:]) if arg is None else parse_args(arg)
    ############
    #  Config  #
    ############
    input_arg["custom_set_train"] = "TAT-Vol1-train.csv"
    input_arg["custom_set_test"] = "TAT-Vol1-eval.csv"
    input_arg["tokenize_config"] = "openai/whisper-base"
    input_arg["model_config"] = "openai/whisper-base"
    input_arg["output_dir"] = "whisper-base_result/"
    input_arg["group_by_length"] = True

    print("input_arg", input_arg)
    # repo_name = f"{input_arg['model_config']}-{input_arg['custom_set_train'] if 'custom_set_train' in input_arg else input_arg['train_subset']}"
    # repo_name = repo_name.replace("/", "_")
    repo_name = f"{input_arg['model_config']}-TAT-Vol1"

    # feature_extractor = WhisperFeatureExtractor.from_pretrained(input_arg['model_config'])
    processor = WhisperProcessor.from_pretrained(input_arg["model_config"], task="transcribe")
    processor.save_pretrained(repo_name)
    model = WhisperForConditionalGeneration.from_pretrained(input_arg["model_config"])  # BUG
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    audio_feature_key = inspect.getfullargspec(model.forward).args[1]
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, audio_feature_key=audio_feature_key)

    # data set
    dataset = load_dataset("csv", data_files=input_arg["custom_set_train"], cache_dir=input_arg["cache_dir"])
    dataset = dataset.filter(lambda e: nlp2.is_file_exist(e["path"]))
    if "custom_set_test" in input_arg:
        dataset_test = load_dataset("csv", data_files=input_arg["custom_set_test"], cache_dir=input_arg["cache_dir"])
        dataset_test = dataset_test.filter(lambda e: nlp2.is_file_exist(e["path"]))
        data_test = dataset_test["train"]
    else:
        dataset = dataset["train"].train_test_split(test_size=0.1)
        data_test = dataset["test"]

    data_train = dataset["train"]
    data_train = data_train.map(
        prepare_dataset_whisper,
        num_proc=input_arg["num_proc"],
        fn_kwargs={"feature_extractor": processor.feature_extractor, "audio_feature_key": audio_feature_key},
    )
    data_test = data_test.map(
        prepare_dataset_whisper,
        num_proc=input_arg["num_proc"],
        fn_kwargs={"feature_extractor": processor.feature_extractor, "audio_feature_key": audio_feature_key},
    )
    # data_train = data_train.map(
    #     prepare_dataset_custom, num_proc=input_arg["num_proc"], fn_kwargs={"audio_feature_key": audio_feature_key}
    # )
    # data_test = data_test.map(
    #     prepare_dataset_custom, num_proc=input_arg["num_proc"], fn_kwargs={"audio_feature_key": audio_feature_key}
    # )

    print("init dataset")
    print("data train", data_train)
    print("data test", data_test)

    if not input_arg.get("load_cache", False):
        print("before filtering audio length")
        print("data train", data_train)
        print("data test", data_test)
        if input_arg.get("max_input_length_in_sec", None):
            max_input_length_in_sec = input_arg["max_input_length_in_sec"]
            min_input_length_in_sec = 1
            data_train = data_train.filter(
                lambda x: min_input_length_in_sec * processor.feature_extractor.sampling_rate
                < x
                < max_input_length_in_sec * processor.feature_extractor.sampling_rate,
                input_columns=["lengths"],
            )
            data_test = data_test.filter(
                lambda x: min_input_length_in_sec * processor.feature_extractor.sampling_rate
                < x
                < max_input_length_in_sec * processor.feature_extractor.sampling_rate,
                input_columns=["lengths"],
            )
        print("after filtering audio length")
        print("data train", data_train)
        print("data test", data_test)

        print("before filtering label length")
        print("data train", data_train)
        print("data test", data_test)
        data_train = data_train.filter(lambda x: x is not None and 0 < len(x), input_columns=["labels"])
        data_test = data_test.filter(lambda x: x is not None and 0 < len(x), input_columns=["labels"])
        print("after filtering label length")
        print("data train", data_train)
        print("data test", data_test)

        print("before encoding dataset")
        print("data train", data_train)
        print("data test", data_test)
        phonemize = input_arg.get("phoneme", False)

        if not input_arg.get("only_eval", False):
            data_train = data_train.map(encode_dataset, fn_kwargs={"processor": processor})
        data_test = data_test.map(encode_dataset, fn_kwargs={"processor": processor})
        print("after encoding dataset")
        print("data train", data_train)
        print("data test", data_test)

        data_train.save_to_disk(f"{repo_name}-train.data")
        data_test.save_to_disk(f"{repo_name}-test.data")
    else:
        data_train.load_from_disk(f"{repo_name}-train.data")
        data_test.load_from_disk(f"{repo_name}-test.data")

    print("finalize dataset")
    print("data train", data_train)
    print("data test", data_test)
    print("train labels", data_train[0]["labels"])
    print("test labels", data_test[0]["labels"])

    if input_arg.get("sweep_split_shard", False):
        shuffled_dataset = data_train.shuffle(seed=42)
        data_train = shuffled_dataset.shard(num_shards=input_arg.get("sweep_split_shard"), index=0)
        data_train = data_train.shard(num_shards=input_arg.get("sweep_split_shard"), index=0)
        data_test = data_train

    if "openai/whisper" in input_arg["model_config"]:
        trainer_class = Seq2SeqTrainer
        trainer_aug_class = Seq2SeqTrainingArguments
    else:
        trainer_class = Trainer
        trainer_aug_class = TrainingArguments
        model.gradient_checkpointing_enable()

    training_args = trainer_aug_class(
        output_dir=input_arg.get("output_dir", repo_name),
        length_column_name="lengths",
        group_by_length=input_arg["group_by_length"],
        per_device_train_batch_size=int(input_arg["batch"]),
        per_device_eval_batch_size=int(input_arg["batch"]),
        gradient_accumulation_steps=int(input_arg["grad_accum"]),
        eval_accumulation_steps=int(input_arg["grad_accum"]),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_steps=input_arg.get("eval_steps", 400),
        eval_steps=input_arg.get("eval_steps", 400),
        ddp_find_unused_parameters=True,
        resume_from_checkpoint=input_arg.get("checkpoint", False),
        overwrite_output_dir=input_arg.get("overwrite_output_dir", False),
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="cer",
        num_train_epochs=input_arg.get("epoch", 60),
        fp16=True,
        logging_steps=input_arg.get("logging_steps", 10),
        learning_rate=input_arg.get("learning_rate", 2.34e-4),
        warmup_steps=input_arg.get("warmup_steps", 100),
        save_total_limit=input_arg.get("save_total_limit", 5),
        push_to_hub=False,
        report_to="all",
    )
    if "openai/whisper" in input_arg["model_config"]:
        training_args.predict_with_generate = True
        training_args.generation_max_length = 225

    def compute_metrics(pred):
        pred_ids = pred.predictions
        pred_ids = [i[i != -100] for i in pred_ids]
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_ids = pred.label_ids
        label_ids = [i[i != -100] for i in label_ids]
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)
        cer = cer_cal(label_str, pred_str)
        wer = wer_cal(label_str, pred_str)
        pred_result = [[l, p, cer_cal([l], [p])] for l, p in zip(label_str, pred_str)]
        nlp2.write_csv(pred_result, "pred.csv")
        # print 10 predict result randomly for debug
        random.shuffle(pred_result)
        print("pred_result")
        print("=================================")
        for i in range(10):
            print(pred_result[i])
        print("=================================")
        return {"cer": cer, "wer": wer}

    trainer = trainer_class(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=data_train,
        eval_dataset=data_test,
        tokenizer=processor.feature_extractor,
    )

    if not input_arg.get("only_eval", False):
        freezing_callback = FreezingCallback(trainer, model, input_arg.get("unfreeze_warmup_steps", 1000))
        trainer.add_callback(freezing_callback)
        trainer.train(input_arg.get("checkpoint", None))
    else:
        trainer.evaluate()


if __name__ == "__main__":
    main()
