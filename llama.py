from llama_cpp import Llama
import zmq

context = zmq.Context()
receiver = context.socket(zmq.PULL)
receiver.bind("tcp://*:5555")
sender = context.socket(zmq.PUSH)
sender.connect("tcp://localhost:5556")

model = Llama(
        # model_path="/path/to/model",
        model_path="models/mistral-7b-instruct-v0.2.Q8_0.gguf",
        n_gpu_layers=-1,
        chat_format="llama-2"
        )

system_prompt = """
<s>[INST] What is your favourite condiment? [/INST]
Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s>
[INST] Do you have mayonnaise recipes? [/INST]
"""

conversation_history = []

prompt = system_prompt
while True:
    message = receiver.recv_string()

    formatted_input = "\n[INST]" + message + "[/INST]"
    prompt += formatted_input

    stream = model(
        prompt=prompt,
        max_tokens=4000,
        stream=True,
        stop=["\n"]
    )

    response = ""
    for output in stream:
        token = output["choices"][0]["text"]
        print(token, end="", flush=True)
        response += token

    conversation_history.append({"role": "assistant", "content": response})
