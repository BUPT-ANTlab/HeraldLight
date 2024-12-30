import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="",
    api_version="",
    azure_endpoint=""
)

deployment_name = 'gpt-4o-mini'
print('Sending a test completion job')

prompt = {
  "messages": [
    {
        "role": "system",
        "content": [
            {
              "type": "text",
              "text": "You are an AI assistant that helps people find information."
            },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "please introduce yourself"
            }
          ]
        }
      ]
    }
  ],
  "temperature": 0.7,
  "top_p": 0.95,
  "max_tokens": 800
}

response = client.chat.completions.create(
  model="gpt4",  # model = "deployment_name".
  messages=prompt,
  max_tokens=2048,
  temperature=0.0
)
# response = requests.post(url, headers=headers, data=json.dumps(data)).json()
# analysis = response['choices'][0]['message']['content']
analysis = response.choices[0].message.content
#print('analysis\n', analysis)
