import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('guesty.csv')

# Rename the index column
data = data.rename(columns={data.columns[0]: 'day'})

# Calculate staff needed per cycle (3 beds per staff, rounded up)
for col in ['cycle1', 'cycle2', 'cycle3', 'cycle4']:
    data[f'{col}_staff'] = np.ceil(data[col] / 3)

# Handle missing values in cycle4 for day15
data['cycle4'].fillna(0, inplace=True)
data['cycle4_staff'].fillna(0, inplace=True)

# Calculate total staff hours per day based on 5-hour cycles
data['total_staff_hours'] = (data['cycle1_staff'] + data['cycle2_staff'] + 
                              data['cycle3_staff'] + data['cycle4_staff']) * 5

# Summary statistics
total_hours_needed = data['total_staff_hours'].sum()
max_beds = max(data['cycle1'].max(), data['cycle2'].max(), data['cycle3'].max(), data['cycle4'].max())
min_beds = min(data['cycle1'].min(), data['cycle2'].min(), data['cycle3'].min(), data['cycle4'].min())
avg_beds = (data['cycle1'].mean() + data['cycle2'].mean() + data['cycle3'].mean() + data['cycle4'].mean()) / 4

print(f"Total staff hours needed across 28 days: {total_hours_needed}")
print(f"Maximum beds at any time: {max_beds}")
print(f"Minimum beds at any time: {min_beds}")
print(f"Average beds per cycle: {avg_beds:.2f}")
print("\nStaff needed each day:")
print(data[['day', 'total_staff_hours']].to_string(index=False))

# Create a visualization of the bed count data
plt.figure(figsize=(12, 6))
data.plot(x='day', y=['cycle1', 'cycle2', 'cycle3', 'cycle4'], kind='line', figsize=(12, 6))
plt.title('Bed Occupancy by Cycle Across 28 Days')
plt.xlabel('Day')
plt.ylabel('Number of Beds')
plt.grid(True)
plt.tight_layout()

# Create a visualization of the staff needed
plt.figure(figsize=(12, 6))
data.plot(x='day', y=['cycle1_staff', 'cycle2_staff', 'cycle3_staff', 'cycle4_staff'], kind='line', figsize=(12, 6))
plt.title('Staff Needed by Cycle Across 28 Days')
plt.xlabel('Day')
plt.ylabel('Number of Staff')
plt.grid(True)
plt.tight_layout()

# Calculate the minimum number of staff needed for the entire period
# based on the maximum concurrent staff needed at any point
max_concurrent_staff = max(
    data['cycle1_staff'].max(),
    data['cycle2_staff'].max(), 
    data['cycle3_staff'].max(),
    data['cycle4_staff'].max()
)

print(f"\nMaximum concurrent staff needed at any time: {max_concurrent_staff}")

# Detailed daily staffing requirements
print("\nDetailed daily staffing requirements:")
staff_details = data[['day', 'cycle1', 'cycle1_staff', 'cycle2', 'cycle2_staff', 
                      'cycle3', 'cycle3_staff', 'cycle4', 'cycle4_staff']]
print(staff_details.to_string(index=False))
