from llama_cpp import Llama
import zmq

context = zmq.Context()
whisper_receiver = context.socket(zmq.PULL)
whisper_receiver.bind("tcp://*:5555")
llama_output_sender = context.socket(zmq.PUSH)
llama_output_sender.connect("tcp://localhost:5556")

message_begin_sender = context.socket(zmq.PUSH)
message_begin_sender.connect("tcp://localhost:5557")


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
    message = whisper_receiver.recv_string()

    formatted_input = "\n[INST]" + message + "[/INST]"
    prompt += formatted_input

    stream = model(
        prompt=prompt,
        max_tokens=4000,
        stream=True,
        stop=["\n"]
    )

    message_begin_sender.send_string("beginning transmission")
    response = ""
    chunk = ""
    for output in stream:
        token = output["choices"][0]["text"]
        print(token, end="", flush=True)
        response += token
        chunk += token

        words = chunk.split()
        chunk_length = 7
        if len(words) > chunk_length:
            # Send the first chunk_length words and keep the rest in the chunk
            llama_output_sender.send_string(" ".join(words[:chunk_length]))
            chunk = " ".join(words[chunk_length:]) + (" " if chunk.endswith(" ") else "")

    # Send any remaining words in the chunk
    if chunk.strip():
        llama_output_sender.send_string(chunk.strip())

    llama_output_sender.send_string("\nend of output")
    print("\nend of output")

    conversation_history.append({"role": "assistant", "content": response})
