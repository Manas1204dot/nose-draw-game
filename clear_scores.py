import json

# Clear the high scores file
with open("high_scores.json", "w") as f:
    json.dump([], f)

print("High scores cleared!") 