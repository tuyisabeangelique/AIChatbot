from flask import Flask, render_template, request, jsonify
from chatbot_logic import get_bot_response, intents, predict_class
from analytics_dashboard import log_chatbot_interaction, analytics
import random
import time

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def get_response_route():
    start_time = time.time()
    user_text = request.args.get('msg')

    # Get intent predictions
    ints = predict_class(user_text)
    
    # Log the interaction for analytics
    confidence = float(ints[0]['probability']) if ints else 0.0
    predicted_intent = ints[0]['intent'] if ints else 'unknown'
    response_time = time.time() - start_time
    
    log_chatbot_interaction(user_text, predicted_intent, confidence, response_time)

    if ints:
        tag = ints[0]['intent']
        # Check if the intent is for a wait time
        if "wait_time" in tag:
            # Simulate a real-time API call by generating a random wait time
            wait_time = random.randint(5, 25)
            return jsonify({"response": f"The current estimated wait time at {tag.replace('wait_time_', '').replace('_', ' ').title()} is about {wait_time} minutes."})

    # If it's not a wait_time intent or no intent is found, get the standard response
    response = get_bot_response(user_text)
    return jsonify({"response": response})

@app.route("/analytics")
def analytics_dashboard():
    """Analytics dashboard route"""
    metrics = analytics.calculate_metrics()
    insights = analytics.get_process_optimization_insights()
    
    # Generate visualizations
    chart_path = analytics.generate_visualizations()
    
    return render_template("analytics.html", 
                         metrics=metrics, 
                         insights=insights,
                         chart_path=chart_path)

if __name__ == "__main__":
    app.run(debug=True) 