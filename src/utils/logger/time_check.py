import time


def second_to_hour(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return "%02d:%02d:%02d" % (h, m, s)


class TimeCheck:
    def __init__(self, total_epoch):
        self._total_epoch = total_epoch
        self._start_time = None
        self._time_per_epoch = None
        self._eta = None
        self._count = None

    def start(self):
        self._start_time = time.time()
        self._count = 0

    def update(self, epoch):
        self._count += 1
        cur_time = time.time()

        self._time_per_epoch = (cur_time - self._start_time) / self._count
        self._eta = (self._total_epoch - epoch) * self._time_per_epoch

    @property
    def time_per_epoch(self):
        return second_to_hour(self._time_per_epoch)

    @property
    def eta(self):
        return second_to_hour(self._eta)
