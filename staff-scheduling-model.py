import pandas as pd
import numpy as np
from collections import defaultdict

# Define the cycles and their time ranges
# Assuming clinic starts at 7am, each cycle is 5 hours
cycle_times = {
    'cycle1': (7, 12),   # 7am - 12pm
    'cycle2': (12, 17),  # 12pm - 5pm
    'cycle3': (17, 22),  # 5pm - 10pm
    'cycle4': (22, 3)    # 10pm - 3am (next day)
}

# Load data
data = pd.read_csv('guesty.csv')
data = data.rename(columns={data.columns[0]: 'day'})
data['cycle4'].fillna(0, inplace=True)  # Fill missing value for day15

# Calculate staff needed per cycle (3 beds per staff, rounded up)
for col in ['cycle1', 'cycle2', 'cycle3', 'cycle4']:
    data[f'{col}_staff'] = np.ceil(data[col] / 3)

# Define possible shift types
shift_types = {
    '6hr': 6,
    '8hr': 8,
    '10hr': 10,
    '12hr': 12
}

# Define possible shift start times (7am to 9pm)
# We don't start shifts after 9pm because the last cycle ends at 3am
shift_start_times = list(range(7, 22))

# Generate all possible shifts
possible_shifts = []
for shift_name, duration in shift_types.items():
    for start_time in shift_start_times:
        end_time = (start_time + duration) % 24
        # Check if the shift is at least 6 hours and covers some part of the clinic's operating hours
        if duration >= 6:
            # Create a shift with its coverage of cycles
            shift = {
                'shift_id': f"{shift_name}_{start_time:02d}",
                'start': start_time,
                'end': end_time,
                'duration': duration,
                'cycles_covered': []
            }
            
            # Determine which cycles this shift covers
            for cycle, (cycle_start, cycle_end) in cycle_times.items():
                # Handle cycle4 which spans overnight
                if cycle == 'cycle4':
                    if start_time <= 3 or start_time >= 22 or end_time > 0:
                        shift['cycles_covered'].append(cycle)
                else:
                    # For cycles during the day
                    cycle_covered = False
                    shift_end = end_time if end_time > start_time else end_time + 24
                    
                    # Check if shift covers the cycle start
                    if start_time <= cycle_start and shift_end > cycle_start:
                        cycle_covered = True
                    # Check if shift starts during the cycle
                    elif start_time >= cycle_start and start_time < cycle_end:
                        cycle_covered = True
                        
                    if cycle_covered:
                        shift['cycles_covered'].append(cycle)
                        
            if shift['cycles_covered']:  # Only add shifts that cover at least one cycle
                possible_shifts.append(shift)

# Create a schedule
num_staff = 7  # Start with a reasonable number of staff
staff_schedule = []

# Initialize staff work hours and days off
staff_hours = {i: 0 for i in range(1, num_staff + 1)}
staff_days_off = {i: set() for i in range(1, num_staff + 1)}
staff_previous_shift = {i: None for i in range(1, num_staff + 1)}

# Assign one day off per week for each staff member
for staff_id in range(1, num_staff + 1):
    # For each staff member, assign one day off for each of the 4 weeks
    for week in range(4):
        day_off = week * 7 + staff_id % 7  # Distribute days off evenly
        if day_off == 0:  # Ensure day_off is between 1-28
            day_off = 7
        staff_days_off[staff_id].add(day_off)

# Create the optimized schedule
# We'll use a greedy algorithm that assigns the most efficient shifts
for day in range(1, 29):
    # Get staff requirements for this day
    day_data = data[data['day'] == f'day{day}']
    if day_data.empty:
        continue
        
    staff_requirements = {
        'cycle1': int(day_data['cycle1_staff'].values[0]),
        'cycle2': int(day_data['cycle2_staff'].values[0]),
        'cycle3': int(day_data['cycle3_staff'].values[0]),
        'cycle4': int(day_data['cycle4_staff'].values[0])
    }
    
    # Track how many staff have been assigned to each cycle
    cycle_staff_assigned = {cycle: 0 for cycle in ['cycle1', 'cycle2', 'cycle3', 'cycle4']}
    
    # Sort staff by hours worked (least to most)
    sorted_staff = sorted(staff_hours.items(), key=lambda x: x[1])
    
    for staff_id, hours in sorted_staff:
        # Skip if this is a day off for this staff member
        if day in staff_days_off[staff_id]:
            continue
            
        # Check if all cycles have enough staff
        all_cycles_covered = all(cycle_staff_assigned[cycle] >= staff_requirements[cycle] 
                                for cycle in ['cycle1', 'cycle2', 'cycle3', 'cycle4'])
        if all_cycles_covered:
            break
            
        # Find the best shift for this staff member
        best_shift = None
        best_shift_score = -1
        
        for shift in possible_shifts:
            # Skip if this shift would exceed max hours
            if staff_hours[staff_id] + shift['duration'] > 234:
                continue
                
            # Check if shift start time is within ±1 hour of previous day
            prev_shift = staff_previous_shift[staff_id]
            if prev_shift and day > 1:
                prev_start = prev_shift['start']
                if abs(shift['start'] - prev_start) > 1:
                    continue
            
            # Calculate how useful this shift is
            shift_score = 0
            cycles_helped = []
            
            for cycle in shift['cycles_covered']:
                if cycle_staff_assigned[cycle] < staff_requirements[cycle]:
                    shift_score += 1
                    cycles_helped.append(cycle)
            
            # Only consider shifts that help with understaffed cycles
            if shift_score > 0 and shift_score > best_shift_score:
                best_shift = shift
                best_shift_score = shift_score
        
        # Assign the best shift if found
        if best_shift:
            # Update staff hours
            staff_hours[staff_id] += best_shift['duration']
            
            # Update cycles covered
            for cycle in best_shift['cycles_covered']:
                if cycle_staff_assigned[cycle] < staff_requirements[cycle]:
                    cycle_staff_assigned[cycle] += 1
            
            # Update previous shift
            staff_previous_shift[staff_id] = best_shift
            
            # Add to schedule
            staff_schedule.append({
                'day': day,
                'staff_id': staff_id,
                'shift': best_shift['shift_id'],
                'start': best_shift['start'],
                'end': best_shift['end'],
                'duration': best_shift['duration'],
                'cycles_covered': best_shift['cycles_covered']
            })

# Convert schedule to DataFrame for easier analysis
schedule_df = pd.DataFrame(staff_schedule)

# Check if any cycles are understaffed
understaffed_days = []
for day in range(1, 29):
    day_data = data[data['day'] == f'day{day}']
    if day_data.empty:
        continue
        
    # Get the schedule for this day
    day_schedule = schedule_df[schedule_df['day'] == day]
    
    # Count staff per cycle
    cycle_staff = defaultdict(int)
    for _, shift in day_schedule.iterrows():
        for cycle in shift['cycles_covered']:
            cycle_staff[cycle] += 1
    
    # Check if any cycle is understaffed
    understaffed = False
    for cycle in ['cycle1', 'cycle2', 'cycle3', 'cycle4']:
        required = int(day_data[f'{cycle}_staff'].values[0])
        assigned = cycle_staff[cycle]
        if assigned < required:
            understaffed = True
            understaffed_days.append((day, cycle, required, assigned))

print(f"Total staff used: {num_staff}")
print(f"Total hours worked: {sum(staff_hours.values())}")
print("\nStaff hours breakdown:")
for staff_id, hours in staff_hours.items():
    print(f"Staff {staff_id}: {hours} hours, Days off: {sorted(staff_days_off[staff_id])}")

if understaffed_days:
    print("\nUnderstaffed cycles:")
    for day, cycle, required, assigned in understaffed_days:
        print(f"Day {day}, {cycle}: Required {required}, Assigned {assigned}")
else:
    print("\nAll staffing requirements met!")

# Detailed daily schedule
print("\nDetailed Schedule:")
# Sort by day and then start time
schedule_df_sorted = schedule_df.sort_values(by=['day', 'start'])
for day in range(1, 29):
    day_schedule = schedule_df_sorted[schedule_df_sorted['day'] == day]
    if not day_schedule.empty:
        print(f"\nDay {day}:")
        for _, shift in day_schedule.iterrows():
            start_str = f"{shift['start']:02d}:00"
            end_hour = shift['end']
            end_str = f"{end_hour:02d}:00"
            print(f"  Staff {shift['staff_id']}: {start_str}-{end_str} ({shift['duration']} hrs), Cycles: {shift['cycles_covered']}")

# Calculate overlaps to ensure proper handoffs
overlaps = []
for day in range(1, 29):
    day_schedule = schedule_df[schedule_df['day'] == day]
    if day_schedule.empty:
        continue
        
    staff_times = []
    for _, shift in day_schedule.iterrows():
        start = shift['start']
        end = shift['end']
        if end < start:  # Overnight shift
            end += 24
        staff_id = shift['staff_id']
        staff_times.append((staff_id, start, end))
    
    # Check each pair of staff for overlaps
    for i, (staff1, start1, end1) in enumerate(staff_times):
        for staff2, start2, end2 in staff_times[i+1:]:
            # Check if shifts overlap by at least 1 hour
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            overlap_hours = overlap_end - overlap_start
            
            if overlap_hours >= 1:
                overlaps.append((day, staff1, staff2, overlap_hours))

print("\nShift Overlaps (for handoffs):")
for day, staff1, staff2, hours in overlaps:
    print(f"Day {day}: Staff {staff1} and {staff2} overlap for {hours} hours")

# Final analysis and optimization metrics
total_possible_hours = num_staff * 234
efficiency = sum(staff_hours.values()) / total_possible_hours * 100
print(f"\nStaffing Efficiency: {efficiency:.2f}%")
print(f"Average hours per staff: {sum(staff_hours.values()) / num_staff:.2f}")

# Create shift pattern visualization
staff_shifts = {staff_id: [''] * 28 for staff_id in range(1, num_staff + 1)}
for _, shift in schedule_df.iterrows():
    day_idx = shift['day'] - 1
    staff_id = shift['staff_id']
    shift_label = f"{shift['start']}-{shift['end']}"
    staff_shifts[staff_id][day_idx] = shift_label

print("\nStaff Shift Patterns (Start-End times):")
for staff_id, shifts in staff_shifts.items():
    pattern = ''
    for day, shift in enumerate(shifts, 1):
        if day in staff_days_off[staff_id]:
            pattern += 'OFF,'
        elif shift:
            pattern += f"{shift},"
        else:
            pattern += '-,'
    print(f"Staff {staff_id}: {pattern}")
