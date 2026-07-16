from pynput import keyboard


def _pynput_hotkey(value):
    aliases = {
        "ctrl": "<ctrl>",
        "control": "<ctrl>",
        "shift": "<shift>",
        "alt": "<alt>",
        "win": "<cmd>",
        "windows": "<cmd>",
    }
    parts = []
    for part in value.split("+"):
        token = part.strip().lower()
        parts.append(aliases.get(token, token))
    return "+".join(parts)


class HotkeyManager:
    def __init__(self, on_record_toggle, on_floating_window):
        self.on_record_toggle = on_record_toggle
        self.on_floating_window = on_floating_window
        self.listener = None

    def start(self, record_toggle, floating_window):
        self.stop()
        mappings = {
            _pynput_hotkey(record_toggle): self.on_record_toggle,
            _pynput_hotkey(floating_window): self.on_floating_window,
        }
        self.listener = keyboard.GlobalHotKeys(mappings)
        self.listener.start()

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

