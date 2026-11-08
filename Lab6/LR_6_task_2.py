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

# Частотні таблиці та таблиці правдоподібності
for attribute in ['Outlook', 'Humidity', 'Wind']:
    print(f"Частотна таблиця {attribute}:")
    print(pd.crosstab(data[attribute], data['Play']))
    print(f"\nТаблиця правдоподібності {attribute}:")
    print(pd.crosstab(data[attribute], data['Play'], normalize='columns').round(3))
    print()

prior = data['Play'].value_counts(normalize=True)
print("P(Yes) =", round(prior['Yes'], 3), " P(No) =", round(prior['No'], 3))


def conditional(attribute, value, play):
    subset = data[data['Play'] == play]
    return (subset[attribute] == value).mean()


# P(Yes|Sunny) та P(No|Sunny)
p_sunny = (data['Outlook'] == 'Sunny').mean()
print("\nP(Yes|Sunny) =", round(conditional('Outlook', 'Sunny', 'Yes') * prior['Yes'] / p_sunny, 3))
print("P(No|Sunny) =", round(conditional('Outlook', 'Sunny', 'No') * prior['No'] / p_sunny, 3))

# Прогноз для прикладу
conditions = {'Outlook': 'Rain', 'Humidity': 'High', 'Wind': 'Weak'}
scores = {}
for play in ['Yes', 'No']:
    score = prior[play]
    for attribute, value in conditions.items():
        score *= conditional(attribute, value, play)
    scores[play] = score

total = sum(scores.values())
print("\nУмови:", conditions)
print(f"Оцінка Yes = {scores['Yes']:.4f}, оцінка No = {scores['No']:.4f}")
print(f"P(Yes) = {scores['Yes'] / total:.2f}, P(No) = {scores['No'] / total:.2f}")
print("Матч відбудеться" if scores['Yes'] > scores['No'] else "Матч не відбудеться")
