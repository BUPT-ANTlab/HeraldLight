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


## 7. Prompt Instance

| HeraldLight Prompt Template|
|---|
|**Intersection Knowledge:**<br> This intersection operates with a four-signal-phase system. The signal phases are defined as follows:<br> **ETWT (East and West Through):** Permits vehicles to proceed straight in both the East and West directions.<br> **NTST (North and South Through):** Permits vehicles to proceed straight in both the North and South directions.<br> **ELWL (East and West Left-Turn):** Permits vehicles to make left turns in both the East and West directions.<br> **NLSL (North and South Left-Turn):** Permits vehicles to make left turns in both the North and South directions.<br><br> **Task Description:**<br> **Task 1: Signal Phase Selection**<br> You will receive the queueing vehicle data for each of the four signal phases. Your task is to select the most urgent phase based on the following criteria:<br><br> 1. **Total Queue Calculation:**<br> &nbsp;&nbsp;&nbsp;&nbsp;- **Empty:** Indicates no vehicles are queued for that phase.<br> &nbsp;&nbsp;&nbsp;&nbsp;- **[num1, num2]:** Represents the number of queued vehicles in two lanes controlled by the phase. For example, ETWT controls one lane in the East and one in the West. Sum **num1** and **num2** to obtain the total queue for the phase.<br> &nbsp;&nbsp;&nbsp;&nbsp;- **[num1]:** Indicates that only one lane has queued vehicles, and the other lane is empty.<br><br> 2. **Phase Comparison:**<br> &nbsp;&nbsp;&nbsp;&nbsp;- Compare the total queue numbers across all four phases.<br> &nbsp;&nbsp;&nbsp;&nbsp;- If queue totals are similar between phases, assess the balance of each phase: A large difference between **num1** and **num2** signifies an imbalance. An imbalanced phase leads to inefficient use of traffic duration, as the signal allows both lanes to proceed simultaneously, potentially wasting time when one lane has significantly fewer queued vehicles.<br><br> 3. **Two Versions of Queueing Set:**<br> &nbsp;&nbsp;&nbsp;&nbsp;- **Herald Version:** Calculates future vehicle movements, effectively representing the queueing situation. It is a highly efficient prediction method, and in most cases, it is recommended to use this version for decision-making.<br> &nbsp;&nbsp;&nbsp;&nbsp;- **Original Version:** Represents the vehicles queueing at the current time step and does not account for vehicles that will be running in the future. Only when the Herald version shows severe imbalance (e.g., the final duration calculated based on Herald Version is too long (**Duration > 30**)) should you analyze if this long duration is reasonable, using the Original version for short-term gains.<br><br> **Task 2: Duration Selection**<br> After selecting the optimal signal phase in Task 1, determine the appropriate traffic duration using the following steps:<br> 1. Identify the larger number between **num1** and **num2** in the selected phase and denote it as A.<br> 2. Calculate the initial duration: **Duration = (A * 3) - 1**.<br> 3. Adjust the duration based on the following rules (if you picked Original version, this step should be skipped):<br> &nbsp;&nbsp;&nbsp;&nbsp;- If **Duration > 20**, then **Duration = Duration - 3**.<br> &nbsp;&nbsp;&nbsp;&nbsp;- If **Duration = 14**, then **Duration = Duration - 2**.<br><br> **Task Details:**<br> The queueing numbers for each phase are provided as follows:<br> - **Herald version:**<br> &nbsp;&nbsp;&nbsp;&nbsp;ETWT: [4, 1], NTST: [1], ELWL: [1], NLSL: [1, 4]<br> - **Original version:**<br> &nbsp;&nbsp;&nbsp;&nbsp;ETWT: [2, 1], NTST: [1], ELWL: [1], NLSL: [1, 3]<br><br> **Requirements:**<br> 1. Identify the Optimal Traffic Signal: Analyze the queueing data to select the most urgent phase.<br> 2. Calculate the Duration: Determine the appropriate duration based on the selected phase.<br> 3. Provide the Final Decision: Present the chosen signal phase and duration.<br> **Selection Constraints:**<br> - Only one signal phase can be selected.<br> - The final answer must be formatted precisely as: &lt;signal&gt;YOUR_CHOICE&lt;/signal&gt;, &lt;duration&gt;YOUR_CHOICE&lt;/duration&gt;<br> - **Example:** &lt;signal&gt;ETWT&lt;/signal&gt;, &lt;duration&gt;5&lt;/duration&gt;<br> - If all signal phases are empty, select any signal phase with a default duration of 5 to keep the intersection operational.<br> - Ensure the duration is within the range of **0 to 40**. Durations outside this range are considered invalid.<br> - Each tag (&lt;signal&gt; and &lt;duration&gt;) should appear only once in the final answer.<br><br> **Important:**<br> **You MUST provide the answer in the specified format:**<br> &lt;signal&gt;YOUR_CHOICE&lt;/signal&gt;,&lt;duration&gt;YOUR_CHOICE&lt;/duration&gt;<br> Any other format will not be accepted. |
