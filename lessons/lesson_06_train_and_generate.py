# =============================================================================
# LESSON 6: Train and Generate Text
# =============================================================================
# What we'll build:
#   - Train the full model from Lesson 5 on a real text dataset
#   - Generate text from the trained model (the robot speaks!)
#
# Steps:
#   1. Load a real dataset (tiny shakespeare or custom robot dialogue)
#   2. Tokenize it with the functions from Lesson 1
#   3. Train using the loop from Lesson 3
#   4. Save the model weights to disk
#   5. Load the weights and generate text token by token
#
# Generation algorithm (greedy / temperature sampling):
#   - Start with a prompt: "hello robot"
#   - Model predicts the next token
#   - Append it, repeat until done
#
# After this lesson:
#   - Export model to ONNX or use llama.cpp for Raspberry Pi inference
#   - Add text-to-speech (pyttsx3 / espeak) for the robot voice
#   - Add speech-to-text (whisper) so the robot can listen
#
# Requires: pip install torch
# =============================================================================

# TODO: coming in Lesson 6
