from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys
import glob
import os

path = sys.argv[1]
event_files = glob.glob(os.path.join(path, "events.out.tfevents.*"))
latest_event_file = max(event_files, key=os.path.getmtime)
event_acc = EventAccumulator(latest_event_file)
event_acc.Reload()

tag = 'environment/blue_score'
try:
    events = event_acc.Scalars(tag)
    print(f"\nTrend for {tag}:")
    # Print every Nth event to get a sense of the trend
    step_size = max(1, len(events) // 10)
    for i in range(0, len(events), step_size):
        e = events[i]
        print(f"  Step {e.step}: {e.value:.4f}")
    # Print the last few
    print("  ... latest 3:")
    for e in events[-3:]:
        print(f"  Step {e.step}: {e.value:.4f}")
except KeyError:
    print(f"Tag {tag} not found.")

tag_ret = 'environment/episode_return'
try:
    events = event_acc.Scalars(tag_ret)
    print(f"\nTrend for {tag_ret}:")
    step_size = max(1, len(events) // 10)
    for i in range(0, len(events), step_size):
        e = events[i]
        print(f"  Step {e.step}: {e.value:.4f}")
    # Print the last few
    print("  ... latest 3:")
    for e in events[-3:]:
        print(f"  Step {e.step}: {e.value:.4f}")
except KeyError:
    pass

