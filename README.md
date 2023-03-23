# Taiwanese-Whisper
This repo aims to fine-tune the Whipser model for Taiwanese recognition.

We used the TAT dataset to fine-tune Whisper. However, since TAT is not open source, interested programmers are encouraged to fine-tune the model on the SuiSiann datasets instead. [See Reference ]

## Prepare TAT dataset

* run scripts/prepare_TAT.py

    e.g.

    ```
    python prepare_TAT.py --TAT_root /storage/speech_dataset/TAT/TAT-Vol1-train
    python prepare_TAT.py --TAT_root /storage/speech_dataset/TAT/TAT-Vol1-eval
    ```

* You should implement your own dataset preparing script if you're not using TAT 台羅數字調. [See TODO]

## Whisper Training
- modify the input_arg of the config section in train.py. (This will be modified after hyperameter/config search. See TODO)
- run train.py

## Whisper Tokenization
An example of whisper tokenization. Currently we don't add language token [See TODO].
```
Input:                 tsu7-tsi2 ti7 to2-ui7?
Decoded w/ special:    <|startoftranscript|><|transcribe|><|notimestamps|>tsu7-tsi2 ti7 to2-ui7?<|endoftext|>
Decoded w/out special: tsu7-tsi2 ti7 to2-ui7?
Are equal:             True
```

## A Super Initial Result
A super prelimanary result of fine-tuning whisper-base on **TAT-Vol1-train split** for 4 epcohs and tested on **TAT-Vol1-eval split**.

The hyperpameters and architecures are not searched at all. [See TODO]

```
{'eval_loss': 0.5776590704917908, 'eval_cer': 0.1885265649136233, 'eval_wer': 0.615572355497351, 'eval_runtime': 469.404, 'eval_samples_per_second': 5.588, 'eval_steps_per_second': 0.699, 'epoch': 4.0}  
```

Some examples:

| Ground Truth | Prediction | CER |
| ------------ | ---------- | --- |
|'tau3 than3 tsit8-kua2 tsinn5 lai5 ka7 tshu3-lai7 sann1-thinn1.'| 'tau1-thann1 tsit8 kua1-tsing5 lai5 ka7 tshu3-lai7 sann1 thinn5.'| 0.16|
|'bun5-tsiong1 it4.'| 'bun5-tsiong1 it4.'|0.0|                                                                                                                                                            
|'na2 hainn1 kua1 na2 tsau2 tng2-khi3 tshu3-lai7.'| 'na2 hai7 kua1,na2 tsau2-tng5-kin1 tshu3-lai5.'| 0.21|                                                                                   
|'u7 tso3 ang5-ku1-kue2.'| 'u7 tso3 ang5-ku7-kue2.'| 0.045|                                                                                                                                 
|'a1-sing1 thiann1-tioh8 in1 a1-pa1 hiah4 tua7-siann1.'| 'a1-sing1 khiann1-tioh8 in1 a1-pah4 hiah4 tua7-siann1.'| 0.058|                                                                    
|'hoo7 in1 e7-tang3 ka7 ti7 hak8-hau7 oh8--tioh8--e5,tsin1 iu2-han7 e5 bo2-gi2-bun5 îuî-tshi5 mai3 tng7-kin1!'| 'hoo7 in1 e7-tang3 ka1-tiann7 au2 oh8-tioh8 e5 tsing1-iu2 an1-ne1 bu2-gi2-bun5 ui5-tshi5 mai3 tio7 king1 .'| 0.299|                                                                                                                                                                     
|'hit4 hang7 si7 tai5-uan5 bi2 bun5-hua3.'| 'hi1-han3 si7 tai5-uan5 li2 bun5-hua3.'| 0.154|                                                                                                  
|'ka1-ti7 ka7 bin7 tsim3 lip8 tsui2--li2 e5 tang5-si5.'| 'kah4 li2 ka7 in1-tsing1 u7-jip8-tui3-li7 e5 tang5-si5.'| 0.385|                                                                    
|'iah8-si7 kong2 tso3 tsit8 kiann7 tai7-tsi3 e7-tang3 sun7-sua3 tso3 ho2 ling7-gua7 tsit8 kiann7 tai7-tsi3.'| 'iah8-si7 kong2 tso3 tsit8-kai2 tsi3 e7-tang3 sun7-sua3 tso3-ho2 tsit8-ka1 tai7-tsi3,e7-tang3 sun7-sua3 tso3-ho2-lin7-gua7 tsit4-kai3-tsi2.'| 0.543|
|'sua3--loh8-lai5 khuann3 gi5-lan5-kuan7 bin5-a2-tsai3 ji7 ho7 e5 thinn1-khi3.'| 'sua3-loh8-lai5 khuann3 gi5-lan5-kuan7 bin5-a2-tsai3 ji7 ho7 e5 khi1-khi3.'| 0.0526|
## TODO 
- [x] Report the prelimanary result
- [x] Report Error Rate
- [x] Add an example of Whisper tokenizing 台羅數字調
- [ ] Check if we should implement a customized tokenizer
- [ ] Check if we should add language tags (see sample)
- [ ] Hyperparameter search for better performance
- [ ] Architecture search (whisper- small, base, large, ...) for better performance
- [ ] Implement **漢羅台文** regonition
- [ ] Implement **台羅** recognition
- [ ] Build a demo with Gradio
- [ ] Refactor the code
## Reference
* ASR-training GitHub repo*: [voidful/asr-training](https://github.com/voidful/asr-training)
* Whisper paper: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
* Whisper GitHub repo:[openai/whisper](https://github.com/openai/whisper)
* Whisper fine-tuning blog: [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper)
* TAT dataset: [FSW-Taiwanese Across Taiwan Corpus](https://sites.google.com/speech.ntut.edu.tw/fsw/home/tat-corpus)
* SuíSiann Dataset: [台灣媠聲(SuíSiann Dataset)](https://suisiann-dataset.ithuan.tw/)

\* Most of the codes are inherited from this repo. Thank you Eric!
