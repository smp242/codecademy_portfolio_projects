import csv
from imghdr import tests

with open('insurance.csv', newline='') as insurance_csv:
    reader = csv.DictReader(insurance_csv)
    headers = reader.fieldnames
    reader_list = list(reader)

print(headers)
#Characteristics
total_records = len(reader_list)

age_list = [int(i['age']) for i in reader_list]
ave_age = round(sum(age_list)/len(age_list), 2)
age_min = min(age_list)
age_max = max(age_list)

sex_list = [i['sex'] for i in reader_list]
sex_female_total = sex_list.count('female')
sex_male_total = sex_list.count('male')
male_percent = round(((sex_male_total / total_records)* 100), 2)
female_percent = round(((sex_female_total/total_records)* 100), 2)

region_list = [i['region'] for i in reader_list]
region_list_northwest = region_list.count('northwest')
region_list_southwest = region_list.count('southwest')
region_list_northeast = region_list.count('northeast')
region_list_southeast = region_list.count('southeast')

bmi_list = [float(i['bmi']) for i in reader_list]
bmi_average = round(sum(bmi_list)/len(bmi_list), 2)
bmi_min = min(bmi_list)
bmi_max = max(bmi_list)

smoker_list = [i['smoker'] for i in reader_list]
smoker_list_binary = [1 if i == 'yes' else 0 for i in smoker_list]
smoker_count = smoker_list.count('yes')
smoker_percent = round(((smoker_count/total_records)* 100), 2)

children_list = [int(i['children']) for i in reader_list]
count_0children = children_list.count(0)
count_1children = children_list.count(1)
count_2children = children_list.count(2)
count_3children = children_list.count(3)

charges_list = [round(float(i['charges']), 2) for i in reader_list]
charge_min = min(charges_list)
charge_max = max(charges_list)

#%%
# bring linear regression from previous project so we can model
# need to def as function the for loop

def ols_linear_regression(dataset, decimals=2):
    n = len(dataset)
    sum_x  = sum(x for x, _ in dataset) # _ ignores this information "I'm not going to use this"
    sum_y  = sum(y for _, y in dataset)
    sum_xx = sum(x*x for x, _ in dataset)
    sum_xy = sum(x*y for x, y in dataset)

    denom = n*sum_xx - sum_x*sum_x
    if denom == 0:
        raise ValueError("Cannot fit line: all x are equal")

    m = (n*sum_xy - sum_x*sum_y) / denom
    b = (sum_y - m*sum_x) / n

    # round here
    return round(m, decimals), round(b, decimals)

def predict_y(m, b, x, decimals=2):
    """Return predicted y for a given x from regression line y = mx + b."""
    y_hat = m * x + b
    return round(y_hat, decimals)

def pearson_r(dataset):
    """Compute Pearson correlation coefficient R for a list of (x, y) tuples."""
    n = len(dataset)
    xs = [x for x, _ in dataset]
    ys = [y for _, y in dataset]

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    # Numerator
    num = sum((x - mean_x) * (y - mean_y) for x, y in dataset)

    # Denominator
    den = (sum((x - mean_x)**2 for x in xs) *
           sum((y - mean_y)**2 for y in ys)) ** 0.5

    return round(num / den, 3)


def r_squared(dataset):  # shows percentage of variance explained by the given variables
    """Compute R^2 for a list of (x, y) tuples."""
    r = pearson_r(dataset)
    return round(r ** 2, 3)

testdatapoints = [(1, 2), (2, 0), (3, 4), (4, 4), (5, 3)]
ols_linear_regression(testdatapoints, decimals=2)
#%%
age_vs_charges = list(zip(age_list, charges_list))
bmi_vs_charges = list(zip(bmi_list, charges_list))
smoker_vs_charges = list(zip(smoker_list_binary, charges_list))

# print(len(age_vs_charges))
m, b = ols_linear_regression(age_vs_charges)
print(f'A linear regression formula suggests that insurance charges increase by ${m} for every year of life, with a y-intercept at ${b}')

print('A 20 year old person will likely pay: ',predict_y(257.72, 3165.89, 20))
print('A 30 year old person will likely pay: ',predict_y(257.72, 3165.89, 30))
print('A 40 year old person will likely pay: ',predict_y(257.72, 3165.89, 40))

print("Pearson's R (age vs charges):", pearson_r(age_vs_charges))
print("R^2 (age vs charges):", r_squared(age_vs_charges))

print("Pearson's R (bmi vs charges):", pearson_r(bmi_vs_charges))
print("R^2 (bmi vs charges):", r_squared(bmi_vs_charges))

m0, b1 = ols_linear_regression(smoker_vs_charges)
print(f'Smoking status is associated with an increase of about ${m0} in predicted insurance charges compared to non-smokers.')
#%%
# Characteristics
print(f'Total records: {total_records}')
print(f'Average age of this sample: {ave_age}.  Youngest age: {age_min}.  Oldest age: {age_max}')
print(f'Average bmi of this sample: {bmi_average}.  Lowest bmi: {bmi_min}.  Highest bmi: {bmi_max}.')
print(f'Male records: {sex_male_total} records or {male_percent}%.  Female: {sex_female_total} or {female_percent}%.')
print(f'Northwest records: {region_list_northwest}.\nNortheast Records: {region_list_northeast}.\nSouthwest records: {region_list_southwest}.\nSoutheast Records {region_list_southeast}.\n')
print(f'Smoker records: {smoker_count} or {smoker_percent}%.')



#%%
# Different approach to navigating data and filtering list of dictionaries

males = [row for row in reader_list if row['sex'] == 'Male']
females = [row for row in reader_list if row['sex'] == 'Female']
smoker = [row for row in reader_list if row['smoker'] == 'Yes']
children_records = [row for row in reader_list if int(row['children']) >= 1]
