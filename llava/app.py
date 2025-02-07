# llava/app.py
from flask import Flask, request, jsonify
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM  # Import your model here

app = Flask(__name__)

# Initialize your model here (make sure it's properly loaded)
model = LlavaLlamaForCausalLM.from_pretrained("path_to_your_model_or_model_name")

@app.route('/')
def home():
    return "Llava Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the user (make sure it's in the correct format)
        data = request.get_json()
        user_message = data.get("message")

        if user_message:
            # Process input with the model
            output = model.generate(user_message)  # Make sure this matches your model's usage
            return jsonify({"response": output})

        return jsonify({"error": "No message provided"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
