from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

app = Flask(__name__, template_folder='html')

model_path = "./fine-tuned-gpt2"  # Update with the path to your fine-tuned model
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    
    # Generates a response
    inputs = tokenizer.encode(user_input, return_tensors='pt')
    # Append the user's input to the beginning of the generated sequence
    eos_token_tensor = torch.tensor([tokenizer.eos_token_id]).unsqueeze(0)
    input_ids = torch.cat([inputs, eos_token_tensor], dim=-1)

    outputs = model.generate(input_ids, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({"response": response})




if __name__ == '__main__':
    app.run(debug=True)
