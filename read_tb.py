from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys

path = sys.argv[1]
event_acc = EventAccumulator(path)
event_acc.Reload()

print(f"Metrics for {path}:")
for tag in event_acc.Tags()['scalars']:
    events = event_acc.Scalars(tag)
    if events:
        last_event = events[-1]
        print(f"  {tag}: {last_event.value:.4f} (step {last_event.step})")
