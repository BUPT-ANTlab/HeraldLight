import os
import requests
import time
import requests
import json
import base64
def gpt_4o_mini(prompt):

  # Configuration
  API_KEY = "c4ec17b41258493fbe85b8c8e8322d37"
  headers = {
      "Content-Type": "application/json",
      "api-key": API_KEY,
  }


  payload = {
    "messages": prompt,
    "temperature": 0.01,
    "top_p": 0.95,
    "max_tokens": 4096
  }
  max_retries = 5
  retry_delay = 60
  ENDPOINT = "https://swedencentralgpt4thedomian.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-15-preview"


  for attempt in range(max_retries):
      try:
          response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=100)
          response.raise_for_status()

          a = response.json()['choices'][0]['message']['content']

          return a

      except requests.exceptions.Timeout:
          print(f"Timeout error occurred on attempt {attempt + 1}. Retrying in {retry_delay} seconds...")

      except requests.exceptions.ConnectionError:
          print(f"Connection error occurred on attempt {attempt + 1}. Retrying in {retry_delay} seconds...")

      except requests.exceptions.HTTPError as http_err:
          print(f"HTTP error occurred on attempt {attempt + 1}: {http_err}. Retrying in {retry_delay} seconds...")

      except requests.exceptions.RequestException as err:
          print(f"An error occurred on attempt {attempt + 1}: {err}. Retrying in {retry_delay} seconds...")

      except ValueError as parse_err:
          print(f"Failed to parse JSON on attempt {attempt + 1}: {parse_err}. Retrying in {retry_delay} seconds...")

      # 等待指定的时间后重试
      if attempt < max_retries - 1:
          time.sleep(retry_delay)
      else:
          print(f"Failed after {max_retries} attempts. Please check your connection or API status.")
          return None

def llama3170B(prompt):
    OPENROUTER_API_KEY = ""
    YOUR_SITE_URL = ""
    YOUR_APP_NAME = "TSCD"
    max_retries = 5
    retry_delay = 60
    #print(prompt[0]['content'][0]['text'])
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": f"{YOUR_SITE_URL}",  # Optional, for including your app on openrouter.ai rankings.
                    "X-Title": f"{YOUR_APP_NAME}",  # Optional. Shows in rankings on openrouter.ai.
                },
                data=json.dumps({
                    "model": "openai/gpt-4o-mini",  # Optional
                    "messages": [
                        {"role": "user", "content": prompt[0]['content'][0]['text']},
                    ],
                    # "top_p": 1,
                    # "temperature": 0.7,
                    # "frequency_penalty": 0,
                    # "presence_penalty": 0,
                    # "repetition_penalty": 1,
                    # "top_k": 0,
                    # "provider": {
                    #     "order": [
                    #         "Fireworks",
                    #     ]
                    # },
                })
            )
            response_data = response.json()
            text_content = response_data['choices'][0]['message']['content']
            #print(text_content)
            return text_content

        except requests.exceptions.Timeout:
            print(f"Timeout error occurred on attempt {attempt + 1}. Retrying in {retry_delay} seconds...")

        except requests.exceptions.ConnectionError:
            print(f"Connection error occurred on attempt {attempt + 1}. Retrying in {retry_delay} seconds...")

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred on attempt {attempt + 1}: {http_err}. Retrying in {retry_delay} seconds...")

        except requests.exceptions.RequestException as err:
            print(f"An error occurred on attempt {attempt + 1}: {err}. Retrying in {retry_delay} seconds...")

        except ValueError as parse_err:
            print(f"Failed to parse JSON on attempt {attempt + 1}: {parse_err}. Retrying in {retry_delay} seconds...")

        if attempt < max_retries - 1:
            time.sleep(retry_delay)
        else:
            print(f"Failed after {max_retries} attempts. Please check your connection or API status.")
            return None

if __name__ == '__main__':
    llama3170B([{'content':[{'text':"""At this intersection, there are three historical decision points that recorded the selected phase information and the resulting number of vehicles passing through. 
These three past states and decisions can serve as a reference for choosing the phase action and its duration for current decisions. 
Please study and understand these historical actions. 
Note: You are not required to mimic the historical actions. They only represent the objective outcome of vehicle flow caused by those decisions. 
Instead, focus on analyzing the number of vehicles released by these actions, and evaluate whether the historically chosen phase actions and durations.There may be:1. newly unrecorded vehicles run and passed the intersection during the duration, 2. The vehicles in Segment 3 didn't have enough time to pass during the duration.So the The passed vehicle number may be not always equal to the total of all segments. 
effectively relieved the pressure on the segments.
Here are three historical decision points :
A crossroad connects two roads: the north-south and east-west. The traffic light is located at the intersection of the two roads. The north-south road is divided into two sections by the intersection: the north and south. Similarly, the east-west road is divided into the east and west. Each section has two lanes: a through and a left-turn lane. Each lane is further divided into three segments. Segment 1 is the closest to the intersection. Segment 2 is in the middle. Segment 3 is the farthest. In a lane, there may be early queued vehicles and approaching vehicles traveling in different segments. Early queued vehicles have arrived at the intersection and await passage permission. Approaching vehicles will arrive at the intersection in the future.

The traffic light has 4 signal phases. Each signal relieves vehicles' flow in the group of two specific lanes. The state of the intersection is listed below. It describes:
- The group of lanes relieving vehicles' flow under each traffic light phase.
- The number of early queued vehicles of the allowed lanes of each signal.
- The number of approaching vehicles in different segments of the allowed lanes of each signal.

Signal: ETWT
Allowed lanes: Eastern and western through lanes
- Early queued: 0 (East), 0 (West), 0 (Total)
- Segment 1: 0 (East), 0 (West), 0 (Total)
- Segment 2: 0 (East), 0 (West), 0 (Total)
- Segment 3: 0 (East), 0 (West), 0 (Total)

Signal: NTST
Allowed lanes: Northern and southern through lanes
- Early queued: 0 (North), 0 (South), 0 (Total)
- Segment 1: 0 (North), 0 (South), 0 (Total)
- Segment 2: 0 (North), 0 (South), 0 (Total)
- Segment 3: 0 (North), 0 (South), 0 (Total)

Signal: ELWL
Allowed lanes: Eastern and western left-turn lanes
- Early queued: 0 (East), 0 (West), 0 (Total)
- Segment 1: 0 (East), 0 (West), 0 (Total)
- Segment 2: 0 (East), 0 (West), 0 (Total)
- Segment 3: 0 (East), 0 (West), 0 (Total)

Signal: NLSL
Allowed lanes: Northern and southern left-turn lanes
- Early queued: 0 (North), 0 (South), 0 (Total)
- Segment 1: 0 (North), 0 (South), 0 (Total)
- Segment 2: 0 (North), 0 (South), 0 (Total)
- Segment 3: 0 (North), 0 (South), 0 (Total)

Please answer:
Which is the most effective traffic signal that will most significantly improve the traffic condition during the next phase, which relieves vehicles' flow of the allowed lanes of the signal?How long the duration will be the most optimal choice? (please choose the duration between 10 and 40)

Note:
The traffic congestion is primarily dictated by the early queued vehicles, with the MOST significant impact. You MUST pay the MOST attention to lanes with long queue lengths. It is NOT URGENT to consider vehicles in distant segments since they are unlikely to reach the intersection soon.

Requirements:
- Let's think step by step.
- You can only choose one of the signals listed above.
- You must follow the following steps to provide your analysis: Step 1: Provide your analysis for identifying the optimal traffic signal. Step 2: Answer your chosen signal.
- Your choice can only be given after finishing the analysis.
- Your choice must be identified by the tag: <signal>YOUR_CHOICE</signal>,<duration>YOUR_CHOICE</duration>.At this intersection, there are three historical decision points that recorded the selected phase information and the resulting number of vehicles passing through. 
These three past states and decisions can serve as a reference for choosing the phase action and its duration for current decisions. 
Please study and understand these historical actions. 
Note: You are not required to mimic the historical actions. They only represent the objective outcome of vehicle flow caused by those decisions. 
Instead, focus on analyzing the number of vehicles released by these actions, and evaluate whether the historically chosen phase actions and durations.There may be:1. newly unrecorded vehicles run and passed the intersection during the duration, 2. The vehicles in Segment 3 didn't have enough time to pass during the duration.So the The passed vehicle number may be not always equal to the total of all segments. 
effectively relieved the pressure on the segments.
Here are three historical decision points :
A crossroad connects two roads: the north-south and east-west. The traffic light is located at the intersection of the two roads. The north-south road is divided into two sections by the intersection: the north and south. Similarly, the east-west road is divided into the east and west. Each section has two lanes: a through and a left-turn lane. Each lane is further divided into three segments. Segment 1 is the closest to the intersection. Segment 2 is in the middle. Segment 3 is the farthest. In a lane, there may be early queued vehicles and approaching vehicles traveling in different segments. Early queued vehicles have arrived at the intersection and await passage permission. Approaching vehicles will arrive at the intersection in the future.

The traffic light has 4 signal phases. Each signal relieves vehicles' flow in the group of two specific lanes. The state of the intersection is listed below. It describes:
- The group of lanes relieving vehicles' flow under each traffic light phase.
- The number of early queued vehicles of the allowed lanes of each signal.
- The number of approaching vehicles in different segments of the allowed lanes of each signal.

Signal: ETWT
Allowed lanes: Eastern and western through lanes
- Early queued: 0 (East), 0 (West), 0 (Total)
- Segment 1: 0 (East), 0 (West), 0 (Total)
- Segment 2: 0 (East), 0 (West), 0 (Total)
- Segment 3: 0 (East), 0 (West), 0 (Total)

Signal: NTST
Allowed lanes: Northern and southern through lanes
- Early queued: 0 (North), 0 (South), 0 (Total)
- Segment 1: 0 (North), 0 (South), 0 (Total)
- Segment 2: 0 (North), 0 (South), 0 (Total)
- Segment 3: 0 (North), 0 (South), 0 (Total)

Signal: ELWL
Allowed lanes: Eastern and western left-turn lanes
- Early queued: 0 (East), 0 (West), 0 (Total)
- Segment 1: 0 (East), 0 (West), 0 (Total)
- Segment 2: 0 (East), 0 (West), 0 (Total)
- Segment 3: 0 (East), 0 (West), 0 (Total)

Signal: NLSL
Allowed lanes: Northern and southern left-turn lanes
- Early queued: 0 (North), 0 (South), 0 (Total)
- Segment 1: 0 (North), 0 (South), 0 (Total)
- Segment 2: 0 (North), 0 (South), 0 (Total)
- Segment 3: 0 (North), 0 (South), 0 (Total)

Please answer:
Which is the most effective traffic signal that will most significantly improve the traffic condition during the next phase, which relieves vehicles' flow of the allowed lanes of the signal?How long the duration will be the most optimal choice? (please choose the duration between 10 and 40)

Note:
The traffic congestion is primarily dictated by the early queued vehicles, with the MOST significant impact. You MUST pay the MOST attention to lanes with long queue lengths. It is NOT URGENT to consider vehicles in distant segments since they are unlikely to reach the intersection soon.

Requirements:
- Let's think step by step.
- You can only choose one of the signals listed above.
- You must follow the following steps to provide your analysis: Step 1: Provide your analysis for identifying the optimal traffic signal. Step 2: Answer your chosen signal.
- Your choice can only be given after finishing the analysis.
- Your choice must be identified by the tag: <signal>YOUR_CHOICE</signal>,<duration>YOUR_CHOICE</duration>."""}]}])


