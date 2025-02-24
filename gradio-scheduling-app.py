import pandas as pd
import numpy as np
import pulp as pl
import matplotlib.pyplot as plt
import gradio as gr
from itertools import product
import io
import base64
import tempfile
import os

def optimize_staffing(
    csv_file,
    beds_per_staff,
    max_hours_per_staff,
    hours_per_cycle,
    rest_days_per_week,
    clinic_start,
    clinic_end,
    overlap_time,
    max_start_time_change
):
    # Load data
    if isinstance(csv_file, str):
        # Handle the case when a filepath is passed directly
        data = pd.read_csv(csv_file)
    else:
        # Handle the case when file object is uploaded through Gradio
        data = pd.read_csv(io.StringIO(csv_file.decode('utf-8')))
    
    # Rename the index column if necessary
    if data.columns[0] not in ['day', 'Day', 'DAY']:
        data = data.rename(columns={data.columns[0]: 'day'})
    
    # Fill missing values
    for col in data.columns:
        if col.startswith('cycle'):
            data[col].fillna(0, inplace=True)
    
    # Calculate clinic hours
    if clinic_end < clinic_start:
        clinic_hours = 24 - clinic_start + clinic_end
    else:
        clinic_hours = clinic_end - clinic_start
    
    # Parameters
    BEDS_PER_STAFF = float(beds_per_staff)
    MAX_HOURS_PER_STAFF = float(max_hours_per_staff)
    HOURS_PER_CYCLE = float(hours_per_cycle)
    REST_DAYS_PER_WEEK = int(rest_days_per_week)
    SHIFT_TYPES = [6, 8, 10, 12]  # Standard shift types
    OVERLAP_TIME = float(overlap_time)
    CLINIC_START = int(clinic_start)
    CLINIC_END = int(clinic_end)
    CLINIC_HOURS = clinic_hours
    MAX_START_TIME_CHANGE = int(max_start_time_change)
    
    # Calculate staff needed per cycle (beds/BEDS_PER_STAFF, rounded up)
    for col in data.columns:
        if col.startswith('cycle') and not col.endswith('_staff'):
            data[f'{col}_staff'] = np.ceil(data[col] / BEDS_PER_STAFF)
    
    # Get cycle names and number of cycles
    cycle_cols = [col for col in data.columns if col.startswith('cycle') and not col.endswith('_staff')]
    num_cycles = len(cycle_cols)
    
    # Define cycle times
    cycle_times = {}
    for i, cycle in enumerate(cycle_cols):
        cycle_start = (CLINIC_START + i * HOURS_PER_CYCLE) % 24
        cycle_end = (CLINIC_START + (i + 1) * HOURS_PER_CYCLE) % 24
        cycle_times[cycle] = (cycle_start, cycle_end)
    
    # Get staff requirements
    max_staff_needed = max([data[f'{cycle}_staff'].max() for cycle in cycle_cols])
    
    # Define possible shift start times
    shift_start_times = list(range(CLINIC_START, CLINIC_START + int(CLINIC_HOURS) - min(SHIFT_TYPES) + 1))
    
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
    
    # Estimate minimum number of staff needed
    total_staff_hours = 0
    for _, row in data.iterrows():
        for cycle in cycle_cols:
            total_staff_hours += row[f'{cycle}_staff'] * HOURS_PER_CYCLE
    
    min_staff_estimate = np.ceil(total_staff_hours / MAX_HOURS_PER_STAFF)
    
    # Get number of days in the dataset
    num_days = len(data)
    
    # Add some buffer for constraints like rest days and shift changes
    estimated_staff = max(min_staff_estimate, max_staff_needed + 1)
    
    def optimize_schedule(num_staff):
        # Create a binary linear programming model
        model = pl.LpProblem("Staff_Scheduling", pl.LpMinimize)
        
        # Decision variables
        # x[s,d,shift] = 1 if staff s works shift on day d
        x = pl.LpVariable.dicts("shift", 
                               [(s, d, shift['id']) for s in range(1, num_staff+1) 
                                                    for d in range(1, num_days+1) 
                                                    for shift in possible_shifts],
                               cat='Binary')
        
        # Objective: Minimize total staff hours while ensuring coverage
        model += pl.lpSum(x[(s, d, shift['id'])] * shift['duration'] 
                         for s in range(1, num_staff+1) 
                         for d in range(1, num_days+1) 
                         for shift in possible_shifts)
        
        # Constraint: Each staff works at most one shift per day
        for s in range(1, num_staff+1):
            for d in range(1, num_days+1):
                model += pl.lpSum(x[(s, d, shift['id'])] for shift in possible_shifts) <= 1
        
        # Constraint: Each staff has at least one rest day per week
        for s in range(1, num_staff+1):
            for w in range((num_days + 6) // 7):  # Number of weeks
                week_start = w*7 + 1
                week_end = min(week_start + 6, num_days)
                model += pl.lpSum(x[(s, d, shift['id'])] 
                                 for d in range(week_start, week_end+1) 
                                 for shift in possible_shifts) <= (week_end - week_start + 1) - REST_DAYS_PER_WEEK
        
        # Constraint: Each staff works at most MAX_HOURS_PER_STAFF in the period
        for s in range(1, num_staff+1):
            model += pl.lpSum(x[(s, d, shift['id'])] * shift['duration'] 
                             for d in range(1, num_days+1) 
                             for shift in possible_shifts) <= MAX_HOURS_PER_STAFF
        
        # Constraint: Each cycle has enough staff each day
        for d in range(1, num_days+1):
            day_index = d - 1  # 0-indexed for DataFrame
            
            for cycle in cycle_cols:
                staff_needed = data.iloc[day_index][f'{cycle}_staff']
                
                # Get all shifts that cover this cycle
                covering_shifts = [shift for shift in possible_shifts if cycle in shift['cycles_covered']]
                
                model += pl.lpSum(x[(s, d, shift['id'])] 
                                 for s in range(1, num_staff+1) 
                                 for shift in covering_shifts) >= staff_needed
        
        # Solve model with a time limit
        model.solve(pl.PULP_CBC_CMD(timeLimit=300, msg=0))
        
        # Check if a feasible solution was found
        if model.status == pl.LpStatusOptimal or model.status == pl.LpStatusNotSolved:
            # Extract the solution
            schedule = []
            for s in range(1, num_staff+1):
                for d in range(1, num_days+1):
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
            return None, None
    
    # Try to solve with estimated number of staff
    staff_count = int(estimated_staff)
    results = f"Trying with {staff_count} staff...\n"
    schedule, objective = optimize_schedule(staff_count)
    
    # If no solution found, increment staff count until a solution is found
    while schedule is None and staff_count < 15:  # Cap at 15 to avoid infinite loop
        staff_count += 1
        results += f"Trying with {staff_count} staff...\n"
        schedule, objective = optimize_schedule(staff_count)
    
    if schedule is None:
        results += "Failed to find a feasible solution. Try relaxing some constraints."
        return results, None, None, None, None
    
    results += f"Optimal solution found with {staff_count} staff\n"
    results += f"Total staff hours: {objective}\n"
    
    # Convert to DataFrame for analysis
    schedule_df = pd.DataFrame(schedule)
    
    # Analyze staff workload
    staff_hours = {}
    for s in range(1, staff_count+1):
        staff_shifts = schedule_df[schedule_df['staff_id'] == s]
        total_hours = staff_shifts['duration'].sum()
        staff_hours[s] = total_hours
    
    results += "\nStaff Hours:\n"
    for staff_id, hours in staff_hours.items():
        utilization = (hours / MAX_HOURS_PER_STAFF) * 100
        results += f"Staff {staff_id}: {hours} hours ({utilization:.1f}% utilization)\n"
    
    avg_utilization = sum(staff_hours.values()) / (staff_count * MAX_HOURS_PER_STAFF) * 100
    results += f"\nAverage staff utilization: {avg_utilization:.1f}%\n"
    
    # Check coverage for each day and cycle
    coverage_check = []
    for d in range(1, num_days+1):
        day_index = d - 1  # 0-indexed for DataFrame
        
        day_schedule = schedule_df[schedule_df['day'] == d]
        
        for cycle in cycle_cols:
            required = data.iloc[day_index][f'{cycle}_staff']
            
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
    results += f"Coverage satisfaction: {satisfaction:.1f}%\n"
    
    if satisfaction < 100:
        results += "Warning: Not all staffing requirements are met!\n"
        unsatisfied = coverage_df[~coverage_df['satisfied']]
        results += unsatisfied.to_string() + "\n"
    
    # Generate detailed schedule report
    detailed_schedule = "Detailed Schedule:\n"
    for d in range(1, num_days+1):
        day_schedule = schedule_df[schedule_df['day'] == d]
        day_schedule = day_schedule.sort_values(['start'])
        
        detailed_schedule += f"\nDay {d}:\n"
        for _, shift in day_schedule.iterrows():
            start_str = f"{shift['start']:02d}:00"
            end_hour = shift['end']
            end_str = f"{end_hour:02d}:00"
            cycles = ", ".join(shift['cycles_covered'])
            detailed_schedule += f"  Staff {shift['staff_id']}: {start_str}-{end_str} ({shift['duration']} hrs), Cycles: {cycles}\n"
    
    # Generate schedule visualization
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Prepare schedule for plotting
    staff_days = {}
    for s in range(1, staff_count+1):
        staff_days[s] = [0] * num_days  # 0 means off duty
    
    for _, shift in schedule_df.iterrows():
        staff_id = shift['staff_id']
        day = shift['day'] - 1  # 0-indexed
        staff_days[staff_id][day] = shift['duration']
    
    # Plot the schedule
    for s, hours in staff_days.items():
        ax.bar(range(1, num_days+1), hours, label=f'Staff {s}')
    
    ax.set_xlabel('Day')
    ax.set_ylabel('Shift Hours')
    ax.set_title('Staff Schedule')
    ax.set_xticks(range(1, num_days+1))
    ax.legend()
    
    # Save the figure to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        plt.savefig(f.name)
        plt.close()
        plot_path = f.name
    
    # Create a Gantt chart for the first week
    gantt_fig, gantt_ax = plt.subplots(figsize=(15, 10))
    
    # Filter for first week
    week_days = min(7, num_days)
    week1_schedule = schedule_df[schedule_df['day'] <= week_days]
    
    # Set up colors for each staff
    colors = plt.cm.tab10.colors
    
    # Sort by staff then day
    week1_schedule = week1_schedule.sort_values(['staff_id', 'day'])
    
    # Plot Gantt chart
    for staff_id in range(1, staff_count+1):
        staff_shifts = week1_schedule[week1_schedule['staff_id'] == staff_id]
        
        y_pos = staff_id
        for _, shift in staff_shifts.iterrows():
            day = shift['day']
            start_hour = shift['start']
            duration = shift['duration']
            
            # Handle overnight shifts
            end_hour = shift['end']
            if end_hour < start_hour:  # Overnight shift
                gantt_ax.broken_barh([(day-1 + start_hour/24, (24-start_hour)/24), 
                                     (day, end_hour/24)], 
                                    (y_pos-0.4, 0.8), 
                                    facecolors=colors[staff_id % len(colors)])
            else:
                gantt_ax.broken_barh([(day-1 + start_hour/24, duration/24)], 
                                    (y_pos-0.4, 0.8), 
                                    facecolors=colors[staff_id % len(colors)])
            
            # Add text label
            gantt_ax.text(day-1 + start_hour/24 + duration/48, y_pos, 
                         f"{start_hour:02d}-{end_hour:02d}", 
                         verticalalignment='center', fontsize=8)
    
    gantt_ax.set_xlabel('Day')
    gantt_ax.set_yticks(range(1, staff_count+1))
    gantt_ax.set_yticklabels([f'Staff {s}' for s in range(1, staff_count+1)])
    gantt_ax.set_xlim(0, week_days)
    gantt_ax.set_title('Staff Schedule (Week 1)')
    gantt_ax.grid(True)
    
    # Save the Gantt chart
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        gantt_fig.savefig(f.name)
        plt.close(gantt_fig)
        gantt