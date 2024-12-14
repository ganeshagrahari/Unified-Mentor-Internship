from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Initialize the transformer model for conversation (GPT-2, or a similar model)
chatbot = pipeline('text-generation', model='gpt2')

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "Message is required!"}), 400

    # Generate a response using the model
    response = chatbot(user_input, max_length=50, num_return_sequences=1)

    return jsonify({"response": response[0]["generated_text"]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
