# A Dual Large Language Models Architecture with Herald Guided Prompts for Parallel Fine Grained Traffic Signal Control

<a id="Introduction"></a>

## 1. Introduction

Leveraging large language models (LLMs) in traffic signal control (TSC) improves optimization efficiency and interpretability compared to traditional reinforcement learning (RL) methods. However, existing LLM-based approaches are limited by fixed time signal durations and are prone to hallucination errors, while RL methods lack robustness in signal timing decisions and suffer from poor generalization. To address these challenges, this paper proposes HeraldLight, a dual LLMs architecture enhanced by Herald guided prompts. The Herald Module extracts contextual information and forecasts queue lengths for each traffic phase based on real-time conditions. The first LLM, LLM-Agent, uses these forecasts to make fine grained traffic signal control, while the second LLM, LLM-Critic, refines LLM-Agent’s outputs, correcting errors and hallucinations. These refined outputs are used for score-based fine-tuning to improve accuracy and robustness. Simulation experiments using CityFlow on real world datasets covering 224 intersections in Jinan (12), Hangzhou (16), and New York (196) demonstrate that HeraldLight outperforms state of the art baselines, achieving a 20.03% reduction in average travel time across all scenarios and a 10.74% reduction in average queue length on the Jinan and Hangzhou scenarios.

![HeraldLight](./docs/overview.png)

<a id="requirements"></a>

## 2. Requirements

* `python>=3.9`
* `cityflow`
* `pandas==1.5.0`
* `numpy==1.26.2`
* `wandb==0.16.5`
* `transformers==4.51.3`
* `vllm==0.8.5.post1`

> **Note:** [`CityFlow`] requires a Linux environment. We run on Ubuntu.

<a id="Usage"></a>

## 3. Herald Module Learns from Scenario

The **Herald Module** automatically collects scenario information and summarizes per-phase signals/queues.

```shell
python run_herald_summarize.py -jinan --traffic_files anon_3_4_jinan_real.json --wandb_name demo --memo exp1
```

<a id="Training"></a>

## 4. Run ChatGPT Data Collection

Use the script below to collect data and interact with CityFlow via most OpenAI-compatible LLM APIs.
Modify the internal parameters to switch models or endpoints as needed.

```shell
python run_gpt.py
```

## 5. LoRA-based Imitation Fine-tuning

**Step 1.** Fine-tuning scripts are located in `./finetune/*`.

```shell
python fintune/finetune_new.py
```

**Step 2.** Merge LoRA weights after fine-tuning:

```shell
python fintune/merge_new_lora.py
```

## 6. Dual LLMs Mechanism

**Data format** for score-based fine-tuning:

```json
{
  "query": "Here is prompt",
  "responses": [
    "Answer 1",
    "Answer 2"
  ],
  "scores": [0.8, 0.2]
}
```

Then run **score-based fine-tuning**:

```shell
python fintune/RankTrainer/Trainer.py
```

