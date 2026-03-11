import time

class RetryManager:
    def __init__(self, max_retries=5, base_delay=1, max_delay=60, backoff_factor=2):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    def get_delay(self, retry_count):
        """Calculate delay in seconds for the given retry count."""
        delay = self.base_delay * (self.backoff_factor ** retry_count)
        return min(delay, self.max_delay)

    def can_retry(self, retry_count):
        return retry_count < self.max_retries

    def wait(self, retry_count):
        """Blocking wait. Returns immediately if retry_count is 0."""
        if retry_count == 0:
            return
        delay = self.get_delay(retry_count - 1)  # delay after (retry_count-1) failures
        time.sleep(delay)