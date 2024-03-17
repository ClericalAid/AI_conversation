from RealtimeTTS import TextToAudioStream, CoquiEngine
import zmq

def receive_from_llama(llama_output_receiver):
    while True:
        message = llama_output_receiver.recv_string()
        print("message", message)

        if message == "\nend of output":
            break

        yield message

def main():
    context = zmq.Context()
    llama_output_receiver = context.socket(zmq.PULL)
    llama_output_receiver.bind("tcp://*:5556")

    message_handling_receiver = context.socket(zmq.PULL)
    message_handling_receiver.bind("tcp://*:5557")

    engine = CoquiEngine(
        voice="/home/chrish/workspace/code/AI_conversation/voices/uk_female_high_voice.wav"
    )
    stream = TextToAudioStream(engine)
    # stream.feed("Hello world! How are you today?")
    # stream.play_async()

    while True:
        message = message_handling_receiver.recv_string()
        if message == "beginning transmission":
            text_stream = receive_from_llama(llama_output_receiver)
            stream.feed(text_stream)
            stream.play_async()
        elif message == "stop transmission":
            stream.stop()

if __name__ == '__main__':
    main()
