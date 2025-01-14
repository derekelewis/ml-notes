import mlx_whisper

speech_file = "./mlx_whisper/assets/ls_test.flac"
# speech_file = "/Users/dlewis/sample.wav"

output = mlx_whisper.transcribe(speech_file)

print(output)
