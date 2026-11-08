import pandas as pd

data = pd.DataFrame(
    [['Sunny', 'High', 'Weak', 'No'],
     ['Sunny', 'High', 'Strong', 'No'],
     ['Overcast', 'High', 'Weak', 'Yes'],
     ['Rain', 'High', 'Weak', 'Yes'],
     ['Rain', 'Normal', 'Weak', 'Yes'],
     ['Rain', 'Normal', 'Strong', 'No'],
     ['Overcast', 'Normal', 'Strong', 'Yes'],
     ['Sunny', 'High', 'Weak', 'No'],
     ['Sunny', 'Normal', 'Weak', 'Yes'],
     ['Rain', 'Normal', 'Weak', 'Yes'],
     ['Sunny', 'Normal', 'Strong', 'Yes'],
     ['Overcast', 'High', 'Strong', 'Yes'],
     ['Overcast', 'Normal', 'Weak', 'Yes'],
     ['Rain', 'High', 'Strong', 'No']],
    columns=['Outlook', 'Humidity', 'Wind', 'Play'])

# Варіант 14
conditions = {'Outlook': 'Sunny', 'Humidity': 'Normal', 'Wind': 'Strong'}
print("Умови:", conditions)

scores = {}
for play in ['Yes', 'No']:
    subset = data[data['Play'] == play]
    score = len(subset) / len(data)
    print(f"\nP({play}) = {len(subset)}/{len(data)}")
    for attribute, value in conditions.items():
        count = (subset[attribute] == value).sum()
        print(f"P({attribute}={value}|{play}) = {count}/{len(subset)}")
        score *= count / len(subset)
    scores[play] = score
    print(f"Оцінка {play} = {score:.4f}")

# Нормалізація
total = sum(scores.values())
p_yes = scores['Yes'] / total
p_no = scores['No'] / total
print(f"\nP(Yes) = {p_yes:.2f}, P(No) = {p_no:.2f}")
print("Матч відбудеться" if p_yes > p_no else "Матч не відбудеться")
