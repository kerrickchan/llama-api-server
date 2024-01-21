from llama_cpp import Llama

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
  # Download the model file first
  model_path="./mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
  n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
  # The number of CPU threads to use, tailor to your system and the resulting performance
  n_threads=12,
  # The number of layers to offload to GPU, if you have GPU acceleration available
  n_gpu_layers=33,
)

# Simple inference example
output = llm(
  "[INST] {prompt} [/INST]",  # Prompt
  max_tokens=512,  # Generate up to 512 tokens
  # Example stop token - not necessarily correct for this specific model! Please check before using.
  stop=["</s>"],
  echo=True        # Whether to echo the prompt
)

# Chat Completion API

# Set chat_format according to the model you are using
llm = Llama(model_path="./mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf", chat_format="llama-2")
res = llm.create_chat_completion(
  messages=[
    { "role": "system", "content": "You are a story writing assistant." },
    {
      "role": "user",
      "content": "Write a story about llamas."
    }
  ]
)

print(res['choices'][0]['message']['content'])
