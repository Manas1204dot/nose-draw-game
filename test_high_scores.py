import json
import time

def add_sample_scores():
    """Add sample high scores to test the system"""
    
    sample_scores = [
        {"name": "Alice", "score": 95.5, "shape": "window", "date": "2024-01-15 14:30"},
        {"name": "Bob", "score": 87.2, "shape": "star", "date": "2024-01-15 13:45"},
        {"name": "Charlie", "score": 92.1, "shape": "house", "date": "2024-01-15 12:20"},
        {"name": "Diana", "score": 89.8, "shape": "flower", "date": "2024-01-15 11:15"},
        {"name": "Eve", "score": 91.3, "shape": "tree", "date": "2024-01-15 10:30"},
        {"name": "Frank", "score": 85.7, "shape": "cloud", "date": "2024-01-15 09:45"},
        {"name": "Grace", "score": 88.9, "shape": "triangle", "date": "2024-01-15 08:20"},
        {"name": "Henry", "score": 86.4, "shape": "window", "date": "2024-01-15 07:15"},
        {"name": "Ivy", "score": 90.2, "shape": "star", "date": "2024-01-15 06:30"},
        {"name": "Jack", "score": 84.1, "shape": "house", "date": "2024-01-15 05:45"}
    ]
    
    # Save sample scores
    with open("high_scores.json", "w") as f:
        json.dump(sample_scores, f, indent=2)
    
    print("✅ Sample high scores added!")
    print("📊 Top 5 scores:")
    for i, score in enumerate(sample_scores[:5]):
        print(f"  {i+1}. {score['name']} - {score['score']}% ({score['shape']})")
    print("\n🎮 Now run the game to see the high scores on the menu!")

if __name__ == "__main__":
    add_sample_scores() 