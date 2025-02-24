import pandas as pd
import numpy as np
import pulp as pl
from itertools import product
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('guesty.csv')
data = data.rename(columns={data.columns[0]: 'day'})
data['cycle4'].fillna(0, inplace=True)  # Fill missing value

# Parameters
BEDS_PER_STAFF = 3
MAX_HOURS_PER_STAFF = 234
HOURS_PER_CYCLE = 5
REST_DAYS_PER_WEEK = 1
SHIFT_TYPES = [6, 8, 10, 12]
OVERLAP_TIME = 1
CLINIC_START = 7  # 7am
CLINIC_END = 3    # 3am next day
CLINIC_HOURS = 20  # Total operating hours
MAX_START_TIME_CHANGE = 1  # Maximum +/- hours shift can start compared to previous day

# Calculate staff needed per cycle (beds/BEDS_PER_STAFF, rounded up)
for col in ['cycle1', 'cycle2', 'cycle3', 'cycle4']:
    data[f'{col}_staff'] = np.ceil(data[col] / BEDS_PER_STAFF)

# Define cycle times
cycle_times = {
    'cycle1': (CLINIC_START, CLINIC_START + HOURS_PER_CYCLE),
    'cycle2': (CLINIC_START + HOURS_PER_CYCLE, CLINIC_START + 2*HOURS_PER_CYCLE),
    'cycle3': (CLINIC_START + 2*HOURS_PER_CYCLE, CLINIC_START + 3*HOURS_PER_CYCLE),
    'cycle4': (CLINIC_START + 3*HOURS_PER_CYCLE, (CLINIC_START + 4*HOURS_PER_CYCLE) % 24)
}

# Get staff requirements
max_staff_needed = max(
    data['cycle1_staff'].max(),
    data['cycle2_staff'].max(),
    data['cycle3_staff'].max(),
    data['cycle4_staff'].max()
)

# Define possible shift start times (7am to 9pm)
shift_start_times = list(range(CLINIC_START, CLINIC_START + CLINIC_HOURS - min(SHIFT_TYPES) + 1))

# Generate all possible shifts
possible_shifts = []
for duration in SHIFT_TYPES:
    for start_time in shift_start_times:
        end_time = (start_time + duration) % 24
        
        # Create a shift with its coverage of cycles
        shift = {
            'id': f"{duration}hr_{start_time:02d}",
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'cycles_covered': set()
        }
        
        # Determine which cycles this shift covers
        for cycle, (cycle_start, cycle_end) in cycle_times.items():
            # Handle overnight cycles
            if cycle_end < cycle_start:  # overnight cycle
                if start_time >= cycle_start or end_time <= cycle_end or (start_time < end_time and end_time > cycle_start):
                    shift['cycles_covered'].add(cycle)
            else:  # normal cycle
                shift_end = end_time if end_time > start_time else end_time + 24
                cycle_end_adj = cycle_end if cycle_end > cycle_start else cycle_end + 24
                
                # Check for overlap
                if not (shift_end <= cycle_start or start_time >= cycle_end_adj):
                    shift['cycles_covered'].add(cycle)
                        
        if shift['cycles_covered']:  # Only add shifts that cover at least one cycle
            possible_shifts.append(shift)

print(f"Generated {len(possible_shifts)} possible shifts")

# Estimate minimum number of staff needed
total_staff_hours = sum(data['cycle1_staff'] + data['cycle2_staff'] + 
                        data['cycle3_staff'] + data['cycle4_staff']) * HOURS_PER_CYCLE
min_staff_estimate = np.ceil(total_staff_hours / MAX_HOURS_PER_STAFF)

# Add some buffer for constraints like rest days and shift changes
estimated_staff = max(min_staff_estimate, max_staff_needed + 1)
print(f"Estimated minimum staff needed: {estimated_staff}")

def optimize_schedule(num_staff):
    # Create a binary linear programming model
    model = pl.LpProblem("Staff_Scheduling", pl.LpMinimize)
    
    # Decision variables
    # x[s,d,shift] = 1 if staff s works shift on day d
    x = pl.LpVariable.dicts("shift", 
                           [(s, d, shift['id']) for s in range(1, num_staff+1) 
                                                for d in range(1, 29) 
                                                for shift in possible_shifts],
                           cat='Binary')
    
    # Objective: Minimize total staff hours while ensuring coverage
    model += pl.lpSum(x[(s, d, shift['id'])] * shift['duration'] 
                     for s in range(1, num_staff+1) 
                     for d in range(1, 29) 
                     for shift in possible_shifts)
    
    # Constraint: Each staff works at most one shift per day
    for s in range(1, num_staff+1):
        for d in range(1, 29):
            model += pl.lpSum(x[(s, d, shift['id'])] for shift in possible_shifts) <= 1
    
    # Constraint: Each staff has at least one rest day per week
    for s in range(1, num_staff+1):
        for w in range(4):  # 4 weeks
            week_start = w*7 + 1
            week_end = min(week_start + 6, 28)
            model += pl.lpSum(x[(s, d, shift['id'])] 
                             for d in range(week_start, week_end+1) 
                             for shift in possible_shifts) <= (week_end - week_start + 1) - REST_DAYS_PER_WEEK
    
    # Constraint: Each staff works at most MAX_HOURS_PER_STAFF in the 28-day period
    for s in range(1, num_staff+1):
        model += pl.lpSum(x[(s, d, shift['id'])] * shift['duration'] 
                         for d in range(1, 29) 
                         for shift in possible_shifts) <= MAX_HOURS_PER_STAFF
    
    # Constraint: Each cycle has enough staff each day
    for d in range(1, 29):
        day_data = data[data['day'] == f'day{d}']
        if day_data.empty:
            continue
            
        for cycle in ['cycle1', 'cycle2', 'cycle3', 'cycle4']:
            staff_needed = int(day_data[f'{cycle}_staff'].values[0])
            
            # Get all shifts that cover this cycle
            covering_shifts = [shift for shift in possible_shifts if cycle in shift['cycles_covered']]
            
            model += pl.lpSum(x[(s, d, shift['id'])] 
                             for s in range(1, num_staff+1) 
                             for shift in covering_shifts) >= staff_needed
    
    # Soft constraint: Try to maintain similar shift start times across days for each staff
    # (We'll handle this post-optimization with a heuristic approach)
    
    # Solve model with a time limit
    model.solve(pl.PULP_CBC_CMD(timeLimit=300))
    
    # Check if a feasible solution was found
    if model.status == pl.LpStatusOptimal or model.status == pl.LpStatusNotSolved:
        print(f"Optimization status: {pl.LpStatus[model.status]}")
        
        # Extract the solution
        schedule = []
        for s in range(1, num_staff+1):
            for d in range(1, 29):
                for shift in possible_shifts:
                    if pl.value(x[(s, d, shift['id'])]) == 1:
                        # Find the shift details
                        shift_details = next((sh for sh in possible_shifts if sh['id'] == shift['id']), None)
                        
                        schedule.append({
                            'staff_id': s,
                            'day': d,
                            'shift_id': shift['id'],
                            'start': shift_details['start'],
                            'end': shift_details['end'],
                            'duration': shift_details['duration'],
                            'cycles_covered': list(shift_details['cycles_covered'])
                        })
        
        return schedule, model.objective.value()
    else:
        print(f"No feasible solution found with {num_staff} staff. Status: {pl.LpStatus[model.status]}")
        return None, None

# Try to solve with estimated number of staff
staff_count = int(estimated_staff)
schedule, objective = optimize_schedule(staff_count)

# If no solution found, increment staff count until a solution is found
while schedule is None and staff_count < 15:  # Cap at 15 to avoid infinite loop
    staff_count += 1
    print(f"Trying with {staff_count} staff...")
    schedule, objective = optimize_schedule(staff_count)

if schedule is None:
    print("Failed to find a feasible solution. Try relaxing some constraints.")
else:
    print(f"Optimal solution found with {staff_count} staff")
    print(f"Total staff hours: {objective}")
    
    # Convert to DataFrame for analysis
    schedule_df = pd.DataFrame(schedule)
    
    # Analyze staff workload
    staff_hours = {}
    for s in range(1, staff_count+1):
        staff_shifts = schedule_df[schedule_df['staff_id'] == s]
        total_hours = staff_shifts['duration'].sum()
        staff_hours[s] = total_hours
        
    print("\nStaff Hours:")
    for staff_id, hours in staff_hours.items():
        utilization = (hours / MAX_HOURS_PER_STAFF) * 100
        print(f"Staff {staff_id}: {hours} hours ({utilization:.1f}% utilization)")
    
    avg_utilization = sum(staff_hours.values()) / (staff_count * MAX_HOURS_PER_STAFF) * 100
    print(f"\nAverage staff utilization: {avg_utilization:.1f}%")
    
    # Check coverage for each day and cycle
    coverage_check = []
    for d in range(1, 29):
        day_data = data[data['day'] == f'day{d}']
        if day_data.empty:
            continue
            
        day_schedule = schedule_df[schedule_df['day'] == d]
        
        for cycle in ['cycle1', 'cycle2', 'cycle3', 'cycle4']:
            required = int(day_data[f'{cycle}_staff'].values[0])
            
            # Count staff covering this cycle
            assigned = sum(1 for _, shift in day_schedule.iterrows() 
                          if cycle in shift['cycles_covered'])
            
            coverage_check.append({
                'day': d,
                'cycle': cycle,
                'required': required,
                'assigned': assigned,
                'satisfied': assigned >= required
            })
    
    coverage_df = pd.DataFrame(coverage_check)
    satisfaction = coverage_df['satisfied'].mean() * 100
    print(f"Coverage satisfaction: {satisfaction:.1f}%")
    
    if satisfaction < 100:
        print("Warning: Not all staffing requirements are met!")
        print(coverage_df[~coverage_df['satisfied']])
    
    # Generate detailed schedule report
    print("\nDetailed Schedule:")
    for d in range(1, 29):
        day_schedule = schedule_df[schedule_df['day'] == d]
        day_schedule = day_schedule.sort_values(['start'])
        
        print(f"\nDay {d}:")
        for _, shift in day_schedule.iterrows():
            start_str = f"{shift['start']:02d}:00"
            end_hour = shift['end']
            end_str = f"{end_hour:02d}:00"
            cycles = ", ".join(shift['cycles_covered'])
            print(f"  Staff {shift['staff_id']}: {start_str}-{end_str} ({shift['duration']} hrs), Cycles: {cycles}")

    # Generate a visual representation of the schedule
    plt.figure(figsize=(15, 10))
    
    # Prepare schedule for plotting
    staff_days = {}
    for s in range(1, staff_count+1):
        staff_days[s] = [0] * 28  # 0 means off duty
    
    for _, shift in schedule_df.iterrows():
        staff_id = shift['staff_id']
        day = shift['day'] - 1  # 0-indexed
        staff_days[staff_id][day] = shift['duration']
    
    # Plot the schedule
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for s, hours in staff_days.items():
        ax.bar(range(1, 29), hours, label=f'Staff {s}')
    
    ax.set_xlabel('Day')
    ax.set_ylabel('Shift Hours')
    ax.set_title('Staff Schedule')
    ax.legend()
    
    # Save the final schedule to CSV
    schedule_df.to_csv('optimized_schedule.csv', index=False)
    print("\nSchedule saved to 'optimized_schedule.csv'")
