# Hybrid speculative decoding in RAG

Hybrid speculative decoding investigates the benefit of a hybrid speculative decoding method using n-gram and EAGLE-3. Such a hybrid is expected to work even better than its components when both components work well. While EAGLE-3 works well generally in English, n-gram is good in very specific use cases. That is why we investigate the performance in the RAG domain, where n-gram can show its full strength. 

Hybrid speculativde decoding achieves:
- 7.3% more accepted tokens $\tau$ for the same token budget

## Contents (mostly copied from EAGLE)

- [Setup & Installation](#setup--installation)
- [EAGLE-3 Weights](#eagle-3-weights)
- [EAGLE Weights](#eagle-weights)
- [Inference](#inference)
  - [With UI](#with-ui)
  - [With Code](#with-code)
- [Train](#train)
  - [Generate Train Data](#generate-train-data)
  - [Train the Auto-regression Head](#train-the-auto-regression-head)
  - [Inference on custom models](#inference-on-custom-models)
- [Evaluation](#evaluation)


## Setup & Installation


```bash
git clone git@github.com:Gianni-Van-de-Velde/hybrid-speculative-decoding.git
cd EAGLE
python -m venv ~/venvs/ea_env
source ~/venvs/ea_env/bin/activate
pip install -r requirements.txt
```
## EAGLE-3 Weights for hybrid speculative decoding

| Base Model            | EAGLE-3 on Hugging Face                                                             |
|-----------------------|-------------------------------------------------------------------------------------|
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)         | [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B)
 



## EAGLE Weights for hybrid speculative decoding with EAGLE-2

*Note:* The current code defaults to using EAGLE-3. If you want to use EAGLE weights, please specify `use_eagle3=False` in `EaModel.from_pretrained`.

 (Compared to EAGLE, EAGLE-2 does not require additional training and uses the same weights.)

| Base Model  | EAGLE on Hugging Face  | \# EAGLE Parameters |
|------|------|------|
| LLaMA3.1-Instruct 8B | [yuhuili/EAGLE-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE-LLaMA3.1-Instruct-8B)| 0.25B |



## Evaluation
You can test the speed of EAGLE on MT-bench using the following command. The models will be downloaded automatically and you may need to input your Hugging Face [Access Tokens](https://huggingface.co/settings/tokens) by ```huggingface-cli login```.
```bash
python -m eagle.evaluation.gen_ea_answer_llama3chat --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B --base-model-path meta-llama/Llama-3.1-8B-Instruct --use_eagle3
```

The above two commands will each generate a .jsonl file that records the generation results and wall time. Then, you can use evaluation/speed.py to calculate the ratio of speeds.


## Reference
For technical details and full experimental results, please check [the paper](TODO). 
```
@inproceedings{TODO,
}
```