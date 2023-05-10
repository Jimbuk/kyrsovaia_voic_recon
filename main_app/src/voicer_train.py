from tflite_support import metadata
import json
import numpy as np
import os
import librosa
import tensorflow as tf


tflite_file = r"./latest.tflite"


def get_labels(model):
    """Returns a list of labels, extracted from the model metadata."""
    displayer = metadata.MetadataDisplayer.with_model_file(model)
    labels_file = displayer.get_packed_associated_file_list()[0]
    labels = displayer.get_associated_file_buffer(labels_file).decode()
    return [line for line in labels.split('\n')]


def get_input_sample_rate(model):
    """Returns the model's expected sample rate, from the model metadata."""
    displayer = metadata.MetadataDisplayer.with_model_file(model)
    metadata_json = json.loads(displayer.get_metadata_json())
    input_tensor_metadata = metadata_json['subgraph_metadata'][0][
        'input_tensor_metadata'][0]
    input_content_props = input_tensor_metadata['content']['content_properties']
    return input_content_props['sample_rate']


# Get a WAV file for inference and list of labels from the model
labels = get_labels(tflite_file)

# Ensure the audio sample fits the model input
interpreter = tf.lite.Interpreter(tflite_file)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1]
sample_rate = get_input_sample_rate(tflite_file)


def predicter(audio):
    audio_data, _ = librosa.load(audio, sr=sample_rate)
    if len(audio_data) < input_size:
        audio_data.resize(input_size)
    audio_data = np.expand_dims(audio_data[:input_size], axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    top_index = np.argmax(output_data[0])
    label = labels[top_index]
    score = output_data[0][top_index]

    return {"class": label, "score": str(score)}, f"Text: {label}, Score: {score}"


