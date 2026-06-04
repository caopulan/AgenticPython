events = []
events.append("before")
a = 1 / 0
events.append(f"after:{a}")
print(events)
