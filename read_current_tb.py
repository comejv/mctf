from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys
import glob
import os

path = sys.argv[1]
# Get the most recent event file in the directory
event_files = glob.glob(os.path.join(path, "events.out.tfevents.*"))
if not event_files:
    print("No event files found yet.")
    sys.exit(0)

# Sort by modification time to get the latest
latest_event_file = max(event_files, key=os.path.getmtime)
print(f"Reading from {latest_event_file}")

event_acc = EventAccumulator(latest_event_file)
event_acc.Reload()

tags_to_check = ['environment/episode_return', 'environment/blue_score', 'losses/entropy', 'learning_rate', 'environment/episode_length', 'SPS']

print("\nCurrent Training Metrics:")
for tag in tags_to_check:
    try:
        events = event_acc.Scalars(tag)
        if events:
            last_event = events[-1]
            print(f"  {tag}: {last_event.value:.4f} (step {last_event.step})")
    except KeyError:
        print(f"  {tag}: Not available yet")
