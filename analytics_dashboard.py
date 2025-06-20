import json
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
from flask import Flask, render_template, jsonify
import os

class ChatbotAnalytics:
    def __init__(self):
        self.interaction_log = []
        self.intent_accuracy = defaultdict(list)
        self.user_satisfaction = []
        self.load_existing_data()
    
    def log_interaction(self, user_input, predicted_intent, confidence, response_time):
        """Log each user interaction for analysis"""
        interaction = {
            'timestamp': datetime.datetime.now().isoformat(),
            'user_input': user_input,
            'predicted_intent': predicted_intent,
            'confidence': confidence,
            'response_time': response_time,
            'session_id': len(self.interaction_log) // 10  # Simple session tracking
        }
        self.interaction_log.append(interaction)
        self.save_data()
    
    def calculate_metrics(self):
        """Calculate key performance metrics"""
        if not self.interaction_log:
            return {}
        
        df = pd.DataFrame(self.interaction_log)
        
        metrics = {
            'total_interactions': len(self.interaction_log),
            'unique_users': df['session_id'].nunique(),
            'avg_confidence': df['confidence'].mean(),
            'avg_response_time': df['response_time'].mean(),
            'most_common_intents': df['predicted_intent'].value_counts().head(5).to_dict(),
            'hourly_usage': df.groupby(df['timestamp'].str[:13]).size().to_dict(),
            'intent_distribution': df['predicted_intent'].value_counts().to_dict()
        }
        
        return metrics
    
    def generate_visualizations(self):
        """Generate charts and graphs for the dashboard"""
        if not self.interaction_log:
            return []
        
        df = pd.DataFrame(self.interaction_log)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Intent Distribution
        intent_counts = df['predicted_intent'].value_counts().head(8)
        axes[0, 0].pie(intent_counts.values, labels=intent_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Most Common User Intents')
        
        # 2. Confidence Distribution
        axes[0, 1].hist(df['confidence'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Confidence Score Distribution')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Response Time Analysis
        axes[1, 0].scatter(df['confidence'], df['response_time'], alpha=0.6)
        axes[1, 0].set_title('Confidence vs Response Time')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Response Time (seconds)')
        
        # 4. Hourly Usage Pattern
        hourly_usage = df.groupby(df['timestamp'].str[11:13]).size()
        axes[1, 1].bar(hourly_usage.index, hourly_usage.values, color='lightgreen')
        axes[1, 1].set_title('Hourly Usage Pattern')
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Number of Interactions')
        
        plt.tight_layout()
        
        # Save the plot
        chart_path = 'static/analytics_chart.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def get_process_optimization_insights(self):
        """Generate insights relevant to process optimization (like PDK development)"""
        if not self.interaction_log:
            return []
        
        df = pd.DataFrame(self.interaction_log)
        
        insights = []
        
        # Performance bottlenecks
        slow_responses = df[df['response_time'] > df['response_time'].quantile(0.9)]
        if len(slow_responses) > 0:
            insights.append({
                'type': 'performance',
                'title': 'Response Time Optimization Needed',
                'description': f'{len(slow_responses)} interactions took longer than {slow_responses["response_time"].mean():.2f}s',
                'recommendation': 'Consider model optimization or caching for frequently asked questions'
            })
        
        # Low confidence patterns
        low_confidence = df[df['confidence'] < 0.5]
        if len(low_confidence) > 0:
            insights.append({
                'type': 'accuracy',
                'title': 'Model Confidence Issues',
                'description': f'{len(low_confidence)} interactions had confidence below 50%',
                'recommendation': 'Review training data and add more examples for unclear intents'
            })
        
        # User behavior patterns
        most_common_inputs = df['user_input'].value_counts().head(3)
        insights.append({
            'type': 'user_behavior',
            'title': 'Most Common User Queries',
            'description': f'Top queries: {", ".join(most_common_inputs.index)}',
            'recommendation': 'Optimize responses for these high-frequency queries'
        })
        
        return insights
    
    def save_data(self):
        """Save analytics data to file"""
        data = {
            'interaction_log': self.interaction_log,
            'last_updated': datetime.datetime.now().isoformat()
        }
        with open('analytics_data.json', 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_existing_data(self):
        """Load existing analytics data"""
        try:
            with open('analytics_data.json', 'r') as f:
                data = json.load(f)
                self.interaction_log = data.get('interaction_log', [])
        except FileNotFoundError:
            self.interaction_log = []

# Initialize analytics
analytics = ChatbotAnalytics()

def log_chatbot_interaction(user_input, predicted_intent, confidence, response_time):
    """Function to be called from the main chatbot to log interactions"""
    analytics.log_interaction(user_input, predicted_intent, confidence, response_time) 