# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
from collections import OrderedDict

import numpy as np


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0
    start_ts = 0.0
    end_ts = 0.0
    tss = []
    # multi-camera
    location = (np.inf, np.inf)

    @property
    def track_ts(self):
        ts = int(self.end_ts - self.start_ts + sum(self.tss))
        if ts / 3600 >= 1:
            fmt_ts = f"{int(ts//3600)}h"
        elif ts / 60 >=1:
            fmt_ts = f"{int(ts//60)}m"
        else:
            fmt_ts = f"{int(ts)}s"
        return fmt_ts

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_long_lost(self):
        self.state = TrackState.LongLost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        BaseTrack._count = 0
